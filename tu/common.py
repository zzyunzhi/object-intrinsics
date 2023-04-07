import os
from tu.loggers.html_table import HTMLTableVisualizer
from tu.loggers.html_helper import BaseHTMLHelper
import logging

logger = logging.getLogger(__name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

vi_helper = BaseHTMLHelper()
vi = HTMLTableVisualizer(os.path.join(ROOT, 'logs/htmls/common'))
vi.begin_html()
# vi_helper.print_url(vi)

logging.basicConfig(level=logging.INFO)
