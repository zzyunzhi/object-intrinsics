import torch
from functools import partial
from ..utils.ema import EMA
from tu.utils.training import get_children_grad_norm, get_children_grad_norm_safe, get_optimizer_lr
from tu.ddp import check_ddp_consistency
from tu.utils.training import process_batch
import time
import torch.distributed as dist
from ..utils.unit_sphere import UnitSphere
from tu.utils.visualize import dump_helper
from ..utils.plot import normalize_batched_tensor
import os
from tu.utils.config import build_from_config
from tu.loggers.utils import setup_vi
from ..utils.checkpoint import CheckpointIO

import logging
logger = logging.getLogger(__name__)


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


MODULE_KEYS = ['generator', 'discriminator', 'mask_discriminator']
MODULE_KEYS_TO_DATA_KEYS = {
    'generator': ['image'],
    'discriminator': ['image'],
    'mask_discriminator': ['mask'],
}


class Trainer:
    def __init__(
            self,
            modules,
            writer, loss_weight,
            loss_modules,
            it=-1,
    ):
        self.modules = modules
        self.module_keys = MODULE_KEYS
        self.module_keys_to_data_keys = MODULE_KEYS_TO_DATA_KEYS
        for k in self.module_keys:
            setattr(self, k, self.modules[k])
            setattr(self, f'opt_{k}', self.modules[f'opt_{k}'])
            setattr(self, f'sch_{k}', self.modules[f'sch_{k}'])

        self.loss_weight = dict()
        for k, v in loss_weight.items():
            if isinstance(v, dict):
                self.loss_weight[k] = build_from_config(v)
            else:
                self.loss_weight[k] = v
        self.loss_modules = {k: build_from_config(v) for k, v in loss_modules.items()}

        self.writer = writer
        self.it = it
        if self.writer is not None:
            self.vi, self.vi_helper = setup_vi(self.writer.get_logdir())
            self.vis_dir = self.writer.get_logdir
            checkpoint_dir = os.path.join(writer.log_dir, 'checkpoints')
        else:
            checkpoint_dir = None
            self.vi = None

        self.checkpoint_io = CheckpointIO(
            checkpoint_dir,
            **self.modules,
        )
        generator_module = self.generator
        if dist.is_initialized():
            generator_module = generator_module.module
        self.light_sphere = UnitSphere(generator_module.resolution, scale=.5).cuda()

    def train_step(self, data):
        self.it += 1

        for k in self.module_keys:
            self.modules[k].train()

        loss_gen = self.train_step_generator(data)
        with torch.no_grad():
            out = self.generator(bs=data['image'].shape[0], it=self.it, data=data, return_raw=False)
        loss_disc = self.train_step_discriminator_core('discriminator', data, {**out['box']['render_out'], 'c2b': out['box']['prior_info']['c2b']})

        with torch.no_grad():
            out = self.generator(bs=data['image'].shape[0], it=self.it, data=data, return_raw=False)
        loss_mask_dist = self.train_step_discriminator_core('mask_discriminator', data, out['box']['render_out'])

        loss = {**loss_gen, **loss_disc, **loss_mask_dist}

        stats = {}
        for k in self.module_keys:
            for kk, v in get_optimizer_lr(self.modules[f'opt_{k}']).items():
                stats[f'lr/opt_{k}/{kk}'] = v

        for k in self.module_keys:
            self.modules[f'sch_{k}'].step()
        return {**loss, **stats}

    def train_step_generator(self, data):
        ret = dict()
        for k in self.module_keys:
            toggle_grad(self.modules[k], requires_grad=k == 'generator')

        self.opt_generator.zero_grad()

        out = self.generator(bs=data['image'].shape[0], it=self.it, data={}, return_raw=False)
        blob = out['box']

        x_fake = torch.cat([blob['render_out'][k] for k in self.module_keys_to_data_keys['discriminator']], dim=-3)
        d_fake = self.discriminator(x_fake, it=self.it)
        d_fake = d_fake[:, :1]
        loss_disc = self.loss_modules['gan'](d_fake, 1)

        mask_x_fake = torch.cat([blob['render_out'][k] for k in self.module_keys_to_data_keys['mask_discriminator']], dim=-3)
        mask_d_fake = self.mask_discriminator(mask_x_fake, it=self.it)
        # mask_d_fake = mask_d_fake[:, :1]
        loss_mask_disc = self.loss_modules['gan'](mask_d_fake, 1)

        loss_final = loss_disc * self.loss_weight['disc_in_gen'] + loss_mask_disc * self.loss_weight['mask_disc_in_gen']

        ret.update({
            'generator/loss': loss_disc,
            'generator/loss_mask': loss_mask_disc,
        })

        for _, blob in out.items():
            for k in blob['loss'].keys():
                loss_extra = blob['loss'][k]
                loss_final += self.loss_weight[k] * loss_extra
                assert f'generator/{k}' not in ret, (k, ret.keys())
                ret[f'generator/{k}'] = loss_extra

        for prefix, blob in out.items():
            for k in blob['stats'].keys():
                ret[f'{prefix}_stats/{k}'] = blob['stats'][k]

        loss_final.backward()
        self.opt_generator.step()

        m = self.generator
        if dist.is_initialized():
            m = m.module
        for k, grad_norm in get_children_grad_norm_safe(m, verbose=self.it == 0).items():
            if grad_norm is None:
                grad_norm = -1
            ret[f'grad_stats/{k}'] = grad_norm

        return ret

    def train_step_discriminator_core(self, module_key: str, real_data, fake_data):
        ret = dict()
        for k in self.module_keys:
            toggle_grad(self.modules[k], requires_grad=k == module_key)

        discriminator = self.modules[module_key]
        opt_discriminator = self.modules[f'opt_{module_key}']
        opt_discriminator.zero_grad()

        x_real = torch.cat([real_data[k] for k in self.module_keys_to_data_keys[module_key]], dim=-3)
        x_real.requires_grad_()
        d_real = discriminator(x_real, it=self.it)
        d_real = d_real[:, :1]
        loss_real = self.loss_modules['gan'](d_real, 1)

        loss_reg = self.loss_modules['reg'](d_real, x_real)

        x_fake = torch.cat([fake_data[k] for k in self.module_keys_to_data_keys[module_key]], dim=-3)
        x_fake.requires_grad_()
        d_fake = discriminator(x_fake, it=self.it)
        if d_fake.size(1) > 1:
            generator_module = self.generator
            if dist.is_initialized():
                generator_module = generator_module.module

            d_fake, d_fake_aux = torch.split(d_fake, (1, generator_module.pose_prior.repr_dim), dim=1)

            pose_target = fake_data['c2b']
            pose_target = generator_module.pose_prior.pose_to_vec_repr(pose_target)
            loss_aux_pose = self.loss_modules['aux_pose'](d_fake_aux, pose_target)
        else:
            loss_aux_pose = 0

        loss_fake = self.loss_modules['gan'](d_fake, 0)

        loss = loss_real + loss_fake + loss_reg * self.loss_weight['reg'] + loss_aux_pose * self.loss_weight['aux_pose'](self.it)
        loss.backward()
        opt_discriminator.step()

        ret.update({
            f'{module_key}/loss': loss_fake + loss_real,
            f'{module_key}/reg': loss_reg,
            f'{module_key}/fake': loss_fake,
            f'{module_key}/real': loss_real,
            f'{module_key}/aux_pose': loss_aux_pose,
            f'{module_key}/fake_sign': d_fake.sign().mean(),
            f'{module_key}/real_sign': d_real.sign().mean(),
        })
        return ret

    def visualize(self, data):
        if self.vi is not None:
            self.vi.title = f'It {self.it}'
            self.vi.end_html()
            self.vi.begin_html()

        for k in self.module_keys:
            self.modules[k].eval()
        with torch.no_grad():
            out = self.generator(bs=data['image'].shape[0], data=data, it=None, return_raw=True)

        self.visualize_core(out['box']['render_out'], prefix='fake')
        self.visualize_core(data, prefix='real')

    def visualize_core(self, maps, prefix=None):
        generator = self.generator
        discriminator = self.discriminator
        if dist.is_initialized():
            generator = generator.module
            discriminator = discriminator.module
        dump = partial(dump_helper, self, prefix=prefix)
        for k in ['image', 'image_comp', 'image_comp_sphere', 'mask', 'background', 'background_sphere_map', 'color_map', 'specular_map', 'no_specular_map', 'shading_map', 'diff_shading_map', 'amb_shading_map']:
            if k in maps:
                dump(k, maps[k])
        if 'normal_map' in maps:
            dump('normal_map', maps['normal_map'].flip(-3) * 0.5 + 0.5)
        for k in ['diff_shading_map', 'amb_shading_map']:
            if k not in maps:
                continue
            dump(k, maps[k])
        if prefix == 'fake':
            dump('z', normalize_batched_tensor(maps['z_map'], xmin=maps['z_min'][:, None, None, None]))
            dump('light', self.light_sphere.render(generator.light)['shading_map'][None])
        dump(f"{self.module_keys_to_data_keys['discriminator'][0]}_aug", discriminator.aug(maps[self.module_keys_to_data_keys['discriminator'][0]]))

    def save_checkpoint(self, overwrite=True, **kwargs):
        if dist.is_initialized():
            rank = dist.get_rank()
            if rank != 0:
                return
        self.checkpoint_io.save('model.pt', it=self.it, **kwargs)
        logger.info('saving checkpoint to model.pt')
        if not overwrite:
            self.checkpoint_io.save(f'it_{self.it:08d}.pt', it=self.it, **kwargs)
            logger.info(f'saving checkpoint to it_{self.it:08d}.pt')

    def load_checkpoint(self, path, strict=False):
        assert strict is False
        # FIXME this is very hacky
        state_dict = torch.load(path)
        logger.info(str(state_dict.keys()))
        state_dict = {k: v for k, v in state_dict.items() if k in self.module_keys or k in ['epoch', 'it']}
        # FIXME some buffer parameters might be overwritten
        unused = self.checkpoint_io.load(state_dict, strict=strict)
        logger.info(str(unused.keys()))
        logger.info(str({k: v for k, v in unused.items() if not k.startswith('ema') and not isinstance(v, dict)}))
        self.it = unused['it']
        logger.info(f'resuming from it: {self.it}')
        return unused


def train_loops(
        eval_every, print_every, visualize_every, checkpoint_every, checkpoint_overwrite,
        cfg, trainer, train_loader, val_loader, max_epoch, max_it, epoch=-1):
    assert max_epoch is None or max_it is not None, f'infinite loop'
    ema_modules = [
        EMA(trainer.generator, beta=beta, m_ema=None)
        for beta in ([0.99, 0.9] if os.getenv('DEBUG') != '1' else [])
    ]
    ema_modules = {str(ema): ema for ema in ema_modules}

    t0b = time.time()
    while True:
        if max_epoch is not None and epoch > max_epoch:
            logger.info(f'final epoch = {epoch}, it ={trainer.it}, exceeding max epoch {max_epoch}')
            return
        if max_it is not None and trainer.it > max_it:
            logger.info(f'final epoch = {epoch}, it ={trainer.it}, exceeding max it {max_it}')
            return
        epoch += 1
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            batch = process_batch(batch)
            if os.getenv('RECON_OBJ') == '1':
                loss = trainer.train_step_recon(batch)
            else:
                loss = trainer.train_step(batch)

            for ema in ema_modules.values():
                ema.update(trainer.it)

            if os.getenv('DEBUG') == '1' and dist.is_initialized() and trainer.it < 10:
                for k in sorted(trainer.module_keys):
                    check_ddp_consistency(trainer.modules[k], ignore_regex=r'(.*\.[^.]+_(avg|ema))')

            if print_every > 0 and trainer.it % print_every == 0:
                info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (epoch, trainer.it, time.time() - t0b)
                for (k, v) in loss.items():
                    info_txt += ', %s: %.4f' % (k, v)
                if trainer.it <= 100:
                    logger.info(info_txt)
                t0b = time.time()

                if trainer.writer is not None:
                    for k, v in loss.items():
                        trainer.writer.add_scalar(k, v, trainer.it)

            if visualize_every > 0 and trainer.it % visualize_every == 0:
                trainer.visualize(batch)

            if checkpoint_every > 0 and trainer.it % checkpoint_every == 0:
                trainer.save_checkpoint(
                    overwrite=checkpoint_overwrite, epoch=epoch, loss=loss,
                    **{k: v.get_state_dict() for k, v in ema_modules.items()},
                )
