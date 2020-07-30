from setuptools import find_packages, setup


setup(name="cfxgb",
      version="1.0.0",
      description="The official implementation of CFXGB : Cascaded Forest and XGBoost Classifier",
      author="Surya Dheeshjith",
      author_email='Surya.Dheeshjith@gmail.com',
      platforms=["any"],
      url="https://github.com/suryadheeshjith/CFXGB",
      packages=["cfxgb", "cfxgb.lib","cfxgb.lib.cascade","cfxgb.lib.estimators","cfxgb.lib.utils"],
      package_dir={"": "src"},
      install_requires=[
            "numpy",
            "pandas",
            "argparse",
            "imblearn",
            "scikit-learn",
            "joblib",
            "psutil",
            "scipy",
            "simplejson"],
        extras_require={
            "testing": ["pytest-cov"]
      }

)