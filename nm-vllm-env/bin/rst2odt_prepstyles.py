#!/home/rshaw/vllm/nm-vllm-env/bin/python3

# $Id: rst2odt_prepstyles.py 8346 2019-08-26 12:11:32Z milde $
# Author: Dave Kuhlman <dkuhlman@rexx.com>
# Copyright: This module has been placed in the public domain.

"""
Fix a word-processor-generated styles.odt for odtwriter use: Drop page size
specifications from styles.xml in STYLE_FILE.odt.
"""

# Author: Michael Schutte <michi@uiae.at>

from __future__ import print_function

from lxml import etree
import sys
import zipfile
from tempfile import mkstemp
import shutil
import os

NAMESPACES = {
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
}


def prepstyle(filename):

    zin = zipfile.ZipFile(filename)
    styles = zin.read("styles.xml")

    root = etree.fromstring(styles)
    for el in root.xpath("//style:page-layout-properties",
                         namespaces=NAMESPACES):
        for attr in el.attrib:
            if attr.startswith("{%s}" % NAMESPACES["fo"]):
                del el.attrib[attr]

    tempname = mkstemp()
    zout = zipfile.ZipFile(os.fdopen(tempname[0], "w"), "w",
                           zipfile.ZIP_DEFLATED)

    for item in zin.infolist():
        if item.filename == "styles.xml":
            zout.writestr(item, etree.tostring(root))
        else:
            zout.writestr(item, zin.read(item.filename))

    zout.close()
    zin.close()
    shutil.move(tempname[1], filename)


def main():
    args = sys.argv[1:]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        print("Usage: %s STYLE_FILE.odt\n" % sys.argv[0], file=sys.stderr)
        sys.exit(1)
    filename = args[0]
    prepstyle(filename)


if __name__ == '__main__':
    main()
