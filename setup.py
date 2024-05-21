from setuptools import find_packages
from distutils.core import setup

install_requires = [
    'numpy==1.23.1', 
    'tensorboard', 
    'matplotlib', 
    'scipy', 
    'tqdm',
    'transformers==4.34.0', 
    'numba', 
    'wandb', 
    'aligo',
    'pyvirtualdisplay', 
    'pillow', 
    'EasyProcess',
    'opencv-python', 
    'imageio[ffmpeg]', 
    'hydra-core==1.3.2',
    'omegaconf', 
    'hydra-joblib-launcher', 
    'opensimplex',
    'ipykernel', 
    'ipywidgets', 
    'torchscale', 
    'pytz',
    'timm', 
    'lion-pytorch==0.1.2', 
    'torchscale==0.3.0', 
    'triton==2.0.0', # need
    'd4rl', # need d4rl to generate data
    'gym',
]

setup(
    name='FCNet',
    version='1.0.0',
    author='Kai Ma, Hengkai Tan, Songming Liu',
    license="Apache 2.0 License",
    packages=find_packages(),
    author_email='mk23@mails.tsinghua.edu.cn, thj23@mails.tsinghua.edu.cn',
    description='Fourier Controller Networks for Real-Time Decision-Making in Embodied Learning',
    install_requires=install_requires,
)
