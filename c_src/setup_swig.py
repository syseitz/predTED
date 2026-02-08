from distutils.core import setup, Extension

predTED_module = Extension('_predTED',
                           sources=['predTED.c', 'predTED.i'],
                           include_dirs=[], 
                           library_dirs=[],  
                           libraries=[],     
                           swig_opts=[])

setup(name='predTED',
      version='1.0',
      ext_modules=[predTED_module],
      py_modules=['predTED'])