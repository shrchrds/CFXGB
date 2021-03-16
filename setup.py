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
            "numpy==1.16.4",
            "pandas==0.24.2",
            "joblib==0.13.2",
            "psutil==5.7.0",
            "scikit-learn==0.21.2",
            "imblearn==0.5.0",
            "scipy==1.2.1"
            ],
        extras_require={
            "testing": ["pytest-cov"]
      }

)
