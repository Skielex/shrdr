import importlib.util
import re
from glob import glob

from setuptools import Extension, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


class LazyCythonize(list):
    def __init__(self, callback):
        self._list, self.callback = None, callback

    def c_list(self):
        if self._list is None:
            self._list = self.callback()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


def extensions():
    from Cython.Build import cythonize

    maxflow_module = Extension(
        "shrdr._shrdr",
        [
            "shrdr/src/_shrdr.pyx",
        ],
        language="c++",
    )
    return cythonize([maxflow_module], compiler_directives={'language_level': "3"})


# Create templates.
def fill_templates(file_name, new_file_name, cap_types, arc_types, node_types):
    with open(file_name, "r") as fh:
        content = fh.read()

    templates = re.search(r'<template>([\s\S]*)</template>', content, re.IGNORECASE).groups()
    filled_templates = []

    for t in templates:
        full_template = []

        for ct in cap_types:
            # Insert CapType.
            t_ct = t.replace('<CapType>', ct)
            # Insert FlowType.
            t_ct = t_ct.replace('<FlowType>', 'FlowInt' if 'Int' in ct else 'FlowFloat')
            for at in arc_types:
                # Insert ArcType.
                t_at = t_ct.replace('<ArcType>', at)
                for nt in node_types:
                    # Insert NodeType.
                    t_nt = t_at.replace('<NodeType>', nt)
                    # Class name.
                    t_nt = t_nt.replace('<ClassNameExt>', ct + at + nt)
                    # Save full template.
                    full_template.append(t_nt)

        if full_template:
            filled_templates.append(full_template)

    new_content = content
    for t, ft in zip(templates, filled_templates):
        # Insert filled.
        new_content = new_content.replace(t, '\n'.join(ft))

    # Remove template elements.
    new_content = new_content.replace('<template>', '')
    new_content = new_content.replace('</template>\n', '')

    with open(new_file_name, 'w') as fh:
        fh.seek(0)
        fh.write(new_content)
        fh.truncate()


# Import types. We can't use normal import during setup, because _shrdr isn't compiled yet.
types_spec = importlib.util.spec_from_file_location("shrdr.types", "shrdr/types.py")
types = importlib.util.module_from_spec(types_spec)
types_spec.loader.exec_module(types)

cap_types = list(types.capacity_types_lookup.values())
arc_types = list(types.arc_index_types_lookup.values())
node_types = list(types.node_index_types_lookup.values())
for file_name in glob('./shrdr/**/*.pyx.template', recursive=True):
    fill_templates(file_name, file_name.replace('.pyx.template', '.pyx'), cap_types, arc_types, node_types)

setup(name="shrdr",
      version="0.1.0",
      author="Niels Jeppesen",
      author_email="niejep@dtu.dk",
      description="A library of fast s-t graph cut algorithms for Python",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Skielex/shrdr",
      packages=["shrdr"],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: C++",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Image Recognition",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
      ],
      ext_modules=LazyCythonize(extensions),
      setup_requires=["Cython"])
