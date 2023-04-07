from .html_table import HTMLTableVisualizer
from .html_helper import BaseHTMLHelper
from .init_logger import _add_coloring_to_emit_ansi
import logging

# All non-Windows platforms are supporting ANSI escapes so we use them
logging.StreamHandler.emit = _add_coloring_to_emit_ansi(
    logging.StreamHandler.emit)
