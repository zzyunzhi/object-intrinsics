import os
import matplotlib.pyplot as plt
from typing import Union
from PIL import Image
from tu.loggers.html_table import HTMLTableVisualizer, HTMLTableColumnDesc
from tu.loggers.utils import print_viscam_url


class BaseHTMLHelper:
    """Handles higher level logic for HTMLTableVisualizer

    """

    @staticmethod
    def dump_table(vis: HTMLTableVisualizer, layout, table_name='default', col_type='auto', row_ind_suffix='',
                   col_names=None):
        n_cols = len(layout[0])

        if col_type == 'auto':
            col_types = []
            for item in layout[0]:
                if isinstance(item, Image.Image):
                    col_types.append('image')
                elif isinstance(item, plt.Figure):
                    col_types.append('figure')
                elif isinstance(item, str):
                    col_types.append('code')
                elif isinstance(item, dict) and 'image' in item:
                    # item contains two keys: 'image' and 'info'
                    assert 'info' in item and len(item.keys()) == 2
                    col_types.append('image')
                elif isinstance(item, dict) and 'video' in item:
                    # assert 'info' in item and len(item.keys()) == 2
                    col_types.append('video')
                else:
                    raise NotImplementedError(type(item))
        else:
            col_types = [col_type] * len(layout[0])

        columns = [HTMLTableColumnDesc(identifier='row', name='row', type='code')]
        if col_names is None:
            col_names = [f'c{col:02d}' for col in range(n_cols)]
        else:
            assert len(col_names) == n_cols, (col_names, n_cols)
        columns += [HTMLTableColumnDesc(identifier=f"c{col:02d}", name=col_names[col],
                                        type=col_types[col]) for col in
                    range(n_cols)]
        with vis.table(table_name, columns):
            for row in layout:
                vis.row(f"r{vis._row_counter:02d}{row_ind_suffix}", *row)

    @staticmethod
    def print_url(vis: Union[HTMLTableVisualizer, str], verbose=True):
        if isinstance(vis, str):
            path = vis
        else:
            path = vis.get_index_filename()

        url = print_viscam_url(path, verbose=verbose)
        return url

    @staticmethod
    def print_button(vis: HTMLTableVisualizer, prev_vis=None, prev_button=None, next_vis=None, next_button=None):
        html = '<p>'
        if prev_vis is not None:
            html += """
                <a href="%s" style="text-decoration: none">&#9194;&emsp;%s</a>
            """ % (BaseHTMLHelper.print_url(prev_vis, verbose=False), prev_button)
        if prev_vis is not None and next_vis is not None:
            html += """
                &emsp;&emsp;&emsp;
            """
        if next_vis is not None:
            html += """
                <a href="%s" style="text-decoration: none">%s&emsp;&#9193;<br></a></p>
            """ % (BaseHTMLHelper.print_url(next_vis, verbose=False), next_button)
        vis.print_raw(html)
