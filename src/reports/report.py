import datetime
from fpdf import FPDF
# import math
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfMerger
import shutil
# import time

from knee_discovery.knee_discovery import calc_degradation_factor

# object_name,original_resolution_width,original_resolution_height,effective_resolution_width,effective_resolution_height,mAP,
#degradation_factor,knee

header_to_readable = {
    'object_name': '',
    'original_resolution_width': 'Width',
    'original_resolution_height': 'Height',
    'effective_resolution_width': 'Effective Width',
    'effective_resolution_height': 'Effective Height',
    'mAP': 'mAP',
    'degradation_factor': '',
    'knee': 'Knee'
    }

knee_type_to_readable = {
    True: 'Yes',
    False: 'No',
    "unknown": 'Unknown'
    }

def get_timestr_file_last_mod(filename):
    file_path = Path(filename)
    mod_time = file_path.stat().st_mtime
    mod_time_dt = datetime.datetime.fromtimestamp(mod_time)
    fractional_seconds = f"{mod_time % 1:.6f}".split(".")[1]
    formatted_time = mod_time_dt.strftime("%Y%m%d%H%M%S")
    final_time = f"{formatted_time}_{fractional_seconds}"
    return final_time

def generate_report(ctxt):
    print("Report Generator started")

    config = ctxt.config
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])
    reports_path = os.path.join(output_top_dir, config['report']['output_subdir'])
    report_path = os.path.join(reports_path, get_timestr_file_last_mod(results_filename))
    results_filename_in_report_path = os.path.join(report_path, os.path.basename(results_filename))
    config_filename = ctxt.config_filename

    if not os.path.exists(results_filename) and not os.path.exists(results_filename_in_report_path):
        print("Report generator could not find results csv file! "
              + "The knee discovery module was either not run, or not run successfully.")
        return

    display_hyperparams = True
    if not os.path.exists(config_filename):
        print("Warning: Report generator could not find configuration file. "
              + "Hyperparameters will not be displayed.")
        display_hyperparams = False
    
    os.makedirs(report_path, exist_ok=True)

    if not os.path.exists(results_filename_in_report_path):
        try:
            shutil.copy2(results_filename, results_filename_in_report_path)
        except PermissionError:
            print("Report Generator does not have permission to access the results csv file! "
                  + f"Please change the permissions on the following file: {results_filename}")
            return
        except IsADirectoryError:
            print(f"Report generator: results csv file {results_filename} "
                  + f"or report destination file {results_filename_in_report_path} is a directory. "
                  + "Please fix and re-run Report Generator.")
            return
        except shutil.SameFileError:
            pass
    
    # Read CSV and calculate degradation factor
    data_IAPC = pd.read_csv(results_filename_in_report_path, index_col=False)
    ## Ensure degradation_factor is calculated
    if 'degradation_factor' not in data_IAPC.columns:
        data_IAPC['data_IAPC'] = calc_degradation_factor(data_IAPC['original_resolution_width'],
                                                                  data_IAPC['original_resolution_height'],
                                                                  data_IAPC['effective_resolution_width'],
                                                                  data_IAPC['effective_resolution_height'])
    
    data_IAPC['median_mAP'] = data_IAPC.groupby('object_name')['mAP'].transform('median')
    sorted_objects = data_IAPC[['object_name', 'median_mAP']].drop_duplicates().sort_values(by='median_mAP', ascending=False)['object_name']

    num_curves_per_graph = 5
    curve_color = ['blue', 'green', 'orange', 'red', 'purple']
    assert len(curve_color) >= num_curves_per_graph
    
    knee_color = 'red'
    graph_array = []
    curve_array = []
    
    for i, object_name in enumerate(sorted_objects):
        if object_name in ctxt.report_names:
            if (i % num_curves_per_graph) == 0 and i != 0:
                graph_array.append(curve_array)
                curve_array = []
            curve_array.append(object_name)
    
    if len(curve_array) > 0:
        graph_array.append(curve_array)

    num_graphs = len(graph_array)

    for g_i, curve_array in enumerate(graph_array):
        plt.figure(figsize=(10, 4))
        
        for o_i, object_name in enumerate(curve_array):
            object_data_IAPC = data_IAPC[data_IAPC['object_name'] == object_name].copy()
            
            # Plot IAP curve
            plt.plot(object_data_IAPC['degradation_factor'], object_data_IAPC['mAP'], label=f"{object_name} IAP Curve", 
                     color=curve_color[o_i], marker='o', markersize=6, markeredgecolor='black', markerfacecolor=curve_color[o_i])
            
            # Mark knee points
            knee_points = object_data_IAPC[object_data_IAPC['knee'] == True]
            if not knee_points.empty:
                plt.scatter(knee_points['degradation_factor'], knee_points['mAP'], color=knee_color, s=100, zorder=5)

        # Finalize and save the plot
        plt.title("IAP Curves with Knee Points")
        plt.xlabel("Degraded Resolution Factor")
        plt.ylabel("Mean Average Precision (mAP)")
        
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Knees', markerfacecolor=knee_color, markersize=10)]
        plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + legend_elements)
        plt.grid(True)
        plt.savefig(os.path.join(report_path, f'iap_curves_{g_i}.pdf'))
        plt.close()

        # Generate the PDF report (same as before)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Times", size=12)  # Set font
        pdf.cell(200, 10, txt="IAP Curve Analysis", ln=True, align='C')
        pdf.ln(10)
        txt = (f"The plot represents {len(curve_array)} IAP curve(s). Each curve corresponds to a specific object detection. "
               "The x-axis shows the degraded resolution factor, by which the original resolution was reduced and then "
               "blown back up, effectively degrading the image quality. The y-axis shows the mean average precision (mAP) "
               "of the model's performance. Red-marked 'knee' points indicate significant inflection points where the mAP "
               "performance starts to plateau.")
        pdf.multi_cell(0, 10, txt=txt)
        pdf.ln(20)

        if display_hyperparams:
            pdf.cell(200, 10, txt="Hyperparameters for the graph", ln=True, align='C')
            pdf.ln(10)
            preprocess_method = config['preprocess_method']
            pdf.cell(90, 10, txt="Image Preprocessing Method", border=1, align='C')
            pdf.cell(90, 10, txt=f"{preprocess_method}", border=1, align='C')
            pdf.ln(10)

            if preprocess_method == 'tiling':
                pdf.cell(90, 10, txt="Stride", border=1, align='C')
                pdf.cell(90, 10, txt=f"{config['preprocess_methods']['tiling']['stride']}", border=1, align='C')
                pdf.ln(10)

            learning_model = config['model']
            pdf.cell(90, 10, txt="Learning Model", border=1, align='C')
            pdf.cell(90, 10, txt=f"{learning_model}", border=1, align='C')
            pdf.ln(10)

        pdf.output(os.path.join(report_path, f'iap_analysis_{g_i}.pdf'))

    # Merge all generated PDFs
    merger = PdfMerger()
    for i in range(num_graphs):
        merger.append(os.path.join(report_path, f'iap_curves_{i}.pdf'))
        merger.append(os.path.join(report_path, f'iap_analysis_{i}.pdf'))
    
    full_report_filename = os.path.join(report_path, 'full_report.pdf')
    merger.write(full_report_filename)
    merger.close()
    
    print(f"Full report generated: {full_report_filename}")


