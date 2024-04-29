import graphviz
from gan_model_generator import ResNet6Generator, ResNetBlock
from gan_model_discriminator import PatchGANDiscriminator
from torchview import draw_graph
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphviz.set_jupyter_format('png')

generator = ResNet6Generator(patch_size=256, use_tanh=False)
discriminator = PatchGANDiscriminator(patch_size=256)

model_graph_g = draw_graph(generator, input_size=(16, 1, 256, 256), device='cpu', show_shapes=True, roll=True, save_graph=True, filename="./final_plots/generator")
model_graph_g.visual_graph

model_graph_d = draw_graph(discriminator, input_size=[(16, 1, 256, 256), (16,1,256,256)], device='cpu', roll=True, show_shapes=True, depth=1, save_graph=True, filename="./final_plots/discriminator")
model_graph_d.visual_graph 

