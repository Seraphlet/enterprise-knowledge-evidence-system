"""Source loading and parsing interfaces."""

from .html import HtmlBlock, HtmlDocument, HtmlSection, HtmlTable, load_html, parse_html

__all__ = [
    "HtmlBlock",
    "HtmlDocument",
    "HtmlSection",
    "HtmlTable",
    "load_html",
    "parse_html",
]

