import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csvs(plot_infos):
    # Create a dictionary to store figures and axes
    figs = {}

    for plot_info in plot_infos:
        # Read the CSV file
        file_path = os.path.join(os.getcwd(), 'final_plots', plot_info['file_path'])
        data = pd.read_csv(file_path)

        # Check if figure already exists, if not create a new one
        if plot_info['figure'] not in figs:
            fig, ax = plt.subplots()
            ax.set_title(plot_info['title'])
            figs[plot_info['figure']] = (fig, ax)

        fig, ax = figs[plot_info['figure']]

        # Plot 'Step' versus 'Value'
        ax.plot(data['Step'], data['Value'], label=plot_info['line_label'], color=plot_info['color'])

        # Set labels
        ax.set_xlabel(plot_info['xlabel'])
        ax.set_ylabel(plot_info['ylabel'])
        ax.legend()
        
        # Add grid
        ax.grid(True)

    # Save each figure to a file in SVG format
    for fig_name, (fig, ax) in figs.items():
        output_path = os.path.join(os.getcwd(), 'final_plots', fig_name)
        fig.savefig(f"{output_path}.svg")

# Usage
# Usage
plot_infos = [
    # fig 1
    # {'file_path': 'gradient_penalty.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Gradient Penalty Value', 'line_label': "Discrimiantor's gradient penalty ", 'title': "Gradient Penalty", 'figure': 'fig1', 'color': 'blue'},
    
    # fig 2
    {'file_path': 'ssim_p2p.csv', 'xlabel': 'Epochs', 'ylabel': 'SSIM Value', 'line_label': "Classical Pix2Pix", 'title': "Structured Similarity Index Between Denoised and Clean Images", 'figure': 'fig2', 'color': 'green'},
    {'file_path': 'ssim_wgan.csv', 'xlabel': 'Epochs', 'ylabel': 'SSIM Value', 'line_label': "Pix2Pix WGAN-GP", 'title': "Structured Similarity Index Between Denoised and Clean Images", 'figure': 'fig2', 'color': 'orange'},
    
    # fig 3
    # {'file_path': 'g_total.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Loss Value', 'line_label': "Generator loss value", 'title': "Generator Loss", 'figure': 'fig3', 'color': 'red'},
    
    # Add more dictionaries for more plots...
    # {'file_path': 'ssim_r2.csv', 'xlabel': 'Epochs', 'ylabel': 'SSIM Value', 'line_label': "Inference steps SSIM value", 'title': "Structured Similarity Index Between Denoised and Clean Images", 'figure': 'fig4', 'color': 'green'},

    # # REMOVED FIGURES
    # {'file_path': 'd_fake.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Loss Value', 'line_label': "Loss on real couples", 'title': "Discriminator Loss", 'figure': 'fig97', 'color': 'blue'},
    # {'file_path': 'd_real.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Loss Value', 'line_label': "Loss on fake couples", 'title': "Discriminator Loss", 'figure': 'fig98', 'color': 'blue'},
    # {'file_path': 'd_total.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Loss Value', 'line_label': "Discriminator loss value", 'title': "Discriminator Loss", 'figure': 'fig99', 'color': 'blue'},
    
    # fig 3
    {'file_path': 'p2pclassica.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Loss Value', 'line_label': "Generator loss value", 'title': "Generator Loss", 'figure': 'fig3', 'color': 'red'},
    {'file_path': 'wgangp.csv', 'xlabel': 'Logging Steps', 'ylabel': 'Loss Value', 'line_label': "Generator loss value", 'title': "Generator Loss", 'figure': 'fig3', 'color': 'blue'},

]
plot_from_csvs(plot_infos)
 