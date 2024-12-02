import datetime
from fpdf import FPDF
import math
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pathlib import Path
from PyPDF2 import PdfMerger
import shutil
# import time

from knee_discovery.knee_discovery import calc_degradation_factor

# object_name,original_resolution_width,original_resolution_height,effective_resolution_width,effective_resolution_height,mAP,
#degradation_factor,knee

header_to_readable = {
    'object_name': 'Object Class',
    'original_resolution_width': 'Width',
    'original_resolution_height': 'Height',
    'effective_resolution_width': 'Effective Width',
    'effective_resolution_height': 'Effective Height',
    'mAP': 'mAP',
    'degradation_factor': '',
    'GSD': 'GSD (meters)',
    'pixels_on_target': 'Pixels on Target',
    'knee': 'Knee'
    }

knee_type_to_readable = {
    True: 'Yes',
    False: 'No',
    "unknown": 'Unknown'
    }

def get_timestr_file_last_mod(filename):
    """
    Generates a timestamp string representing the last modification time of a file, including fractional seconds for precision.

    Args:
        filename (str): The name or path of the file whose modification time is to be retrieved.

    Returns:
        str: A formatted string representing the file's last modification time, including fractional seconds (e.g., `20231110123456_123456`).
    """

    file_path = Path(filename)
    mod_time = file_path.stat().st_mtime
    mod_time_dt = datetime.datetime.fromtimestamp(mod_time)
    fractional_seconds = f"{mod_time % 1:.6f}".split(".")[1]
    formatted_time = mod_time_dt.strftime("%Y%m%d%H%M%S")
    final_time = f"{formatted_time}_{fractional_seconds}"
    return final_time

def display_table(df, intro_text, page_units_used, precision_dict, output_filename):
    # page_units_used = 0
    if df is None or df.shape[0] == 0:
        print("df is empty")
        return None
    cell_height = 5
    # num_lines = df.shape[0]
    pdf = FPDF()

    pdf.set_fill_color(200, 220, 255)

    # page_units_used += (cell_height * num_lines)
    # if page_units_used >= 250:
    pdf.add_page()
    # page_units_used = (cell_height * num_lines)

    pdf.set_font("Times", size=11) # Times Roman
    pdf.ln(20)
    pdf.cell(190, 10, txt=intro_text, ln=True, align='C')
    pdf.ln(4)
    width_dict = {}
    df_temp = df.copy()
    print(df_temp['GSD'])
    for cell_header in df_temp:
        if is_numeric_dtype(df_temp[cell_header]):
            mult = 3
        else:
            mult = 2
        width_header = 2*len(header_to_readable[cell_header])
        print(width_header)
        df_temp[cell_header] = df_temp[cell_header].astype(str)
        print(df_temp)
        width_column = mult*df_temp[cell_header].apply(len).max()
        print(header_to_readable[cell_header], width_header, width_column)
        width_dict[cell_header] = max(width_column, width_header)
    print("width_dict:")
    print(width_dict)
    total_width = sum(list(width_dict.values()))
    print(f"total_width {total_width}")
    for cell_header in width_dict:
        width_dict[cell_header] = int(round(((width_dict[cell_header] * 190) / total_width), 0))
    for cell_header in df:
        # this_header = header_to_readable[cell_header]
        # cell_header_width = 2*len(this_header)
        # cell_max_value_width = 2*max(list(width_dict[cell_header]))
        # cell_width = max(cell_header_width, cell_max_value_width)
        pdf.cell(width_dict[cell_header], cell_height, txt=header_to_readable[cell_header], border=1, align='C', fill=True)
    pdf.ln(cell_height)
    for idx, row in df.iterrows():
        for cell_header in df:
            # this_header = header_to_readable[cell_header]
            this_valuestr = None
            if cell_header in precision_dict:
                this_valuestr = f"{row[cell_header]:.{precision_dict[cell_header]}f}"
            else:
                this_valuestr = str(row[cell_header])
            # if cell_header == 'mAP':
            #     this_valuestr = f"{row['mAP']:.{3}f}"
            # elif cell_header == 'GSD':
            #     this_valuestr = f"{row['GSD']:.{4}f}"
            # else:
            #     this_valuestr = f"{row[cell_header]}"
            pdf.cell(width_dict[cell_header], cell_height, txt=this_valuestr, border=1, align='C')
        # Move to the next line after each row
        pdf.ln(cell_height)
    pdf.output(output_filename)
    return output_filename

def generate_report(ctxt):
    """
    Generates a report based on the results from the knee discovery module and saves it in the configured directory.

    Args:
        ctxt: Context object containing configuration, file paths, and methods for accessing output directories.

    Returns:
        None: Creates and saves a PDF report based on results data and configuration settings.
    """

    config = ctxt.config
    output_top_dir = ctxt.get_output_dir_path()

    reports_path = os.path.join(output_top_dir, config['report']['output_subdir'])

    if 'clean_subdir' in config['report'] and config['report']['clean_subdir']:
        shutil.rmtree(reports_path)

    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])

    report_path = os.path.join(reports_path, get_timestr_file_last_mod(results_filename))

    results_filename_in_report_path = os.path.join(report_path, os.path.basename(results_filename))

    hyperparams_path = os.path.join(output_top_dir, config['train']['output_subdir'])
    hyperparams_filename = os.path.join(hyperparams_path, config['train']['hyperparameters_filename'])
    hyperparams_filename_in_report_path = os.path.join(report_path, os.path.basename(hyperparams_filename))

    if not os.path.exists(results_filename) and not os.path.exists(results_filename_in_report_path):
        print("Error: Report generator could not find results csv file! "
              + "The knee discovery module was either not run, or not run successfully.")
        return

    os.makedirs(report_path, exist_ok=True)

    for filepath in os.listdir(results_path):
        res_filename = os.path.join(results_path, filepath)
        res_filename_in_report_path = os.path.join(report_path, filepath)
        if os.path.isfile(res_filename):
            try:
                shutil.copy2(res_filename, res_filename_in_report_path)
            except PermissionError:
                print(f"Error: Report Generator does not have permission to access {res_filename}!")
                return
            except IsADirectoryError:
                print(f"Error: Report Generator: results csv file {results_filename} "
                      + f"or report destination file {results_filename_in_report_path} is a directory. "
                      + "Please fix and re-run Report Generator.")
                return
            except shutil.SameFileError:
                # this really shouldn't happen due to the if statement above, but if it does, it's perfectly okay
                pass
        else:
            try:
                shutil.copytree(res_filename, res_filename_in_report_path, dirs_exist_ok=True)
            except PermissionError:
                print(f"Error: Report Generator does not have permission to access {res_filename}!")
                return


    if not os.path.exists(hyperparams_filename_in_report_path):
        try:
            shutil.copy2(hyperparams_filename, hyperparams_filename_in_report_path)
        except PermissionError:
            print("Error: Report Generator does not have permission to access the hyperparameters csv file! "
                  + f"Please change the permissions on the following file: {hyperparams_filename}")
            return
        except IsADirectoryError:
            print(f"Report Generator: hyperparameters csv file {hyperparams_filename} "
                  + f"or report destination file {hyperparams_filename_in_report_path} is a directory. "
                  + "Please fix and re-run Report Generator.")
            return
        except shutil.SameFileError:
            # this really shouldn't happen due to the if statement above, but if it does, it's perfectly okay
            pass

    data_IRPC = pd.read_csv(results_filename_in_report_path, index_col=False)
    # data_IRPC['mAP'] = data_IRPC['mAP'].astype(float).apply(lambda x: round(x, 3))
    if 'display_labels' in config['report']:
        report_labels = config['report']['display_labels']
        for label in report_labels:
            if label not in config['target_labels'].values():
                print(f"Warning: Report Generator: label {label} specified in reports_label list was not in target_labels "
                      + "in configuration file.")
        report_labels_filter = [label for label in report_labels if label in config['target_labels'].values()]
        filter_pattern = '|'.join(report_labels_filter)
        data_IRPC = data_IRPC[data_IRPC['object_name'].str.contains(filter_pattern, case=False, na=False)]\
            .reset_index(drop=True).copy()
    data_IRPC['degradation_factor'] = calc_degradation_factor(data_IRPC['original_resolution_width'],
                                                              data_IRPC['original_resolution_height'],
                                                              data_IRPC['effective_resolution_width'],
                                                              data_IRPC['effective_resolution_height'])

    data_IRPC_knee = data_IRPC[data_IRPC['knee'] == True]
    data_IRPC_knee = data_IRPC_knee.sort_values('mAP', ascending=False).reset_index(drop=True)
    precision_dict = {}
    data_IRPC_knee['mAP'] = data_IRPC_knee['mAP'].apply(lambda x: round(x, 3))
    precision_dict['mAP'] = 3
    data_IRPC_knee['GSD'] = data_IRPC_knee['GSD'].apply(lambda x: round(x, 2))
    precision_dict['GSD'] = 2
    data_IRPC_knee = data_IRPC_knee[['object_name', 'mAP', 'GSD', 'pixels_on_target']]
    obj_class_filename = display_table(
        data_IRPC_knee,
        "Mean Average Precision (mAP), Ground Sample Distance (GSD), and Pixels On Target for detection of all classes at knee",
        0, precision_dict, os.path.join(report_path, 'obj_class_table.pdf'))

    display_hyperparams = False
    if not os.path.exists(hyperparams_filename):
        print("Warning: Report generator could not find hyperparameters file. "
              + "Hyperparameters will not be displayed.")
    else:
        pdf = FPDF()

        pdf.add_page()
        pdf.set_font("Times", size=11) # Times Roman
        pdf.ln(20)
        pdf.cell(200, 6, txt="Hyperparameters used", ln=True, align='C')
        pdf.ln(4)

        cell_width = 100
        cell_height = 5

        preprocess_method = config['preprocess_method']
        pdf.cell(cell_width, cell_height, txt="Image Preprocessing Method", border=1, align='C')
        pdf.cell(cell_width, cell_height, txt=f"{preprocess_method}", border=1, align='C')
        pdf.ln(cell_height)

        pdf.cell(cell_width, cell_height, txt="Image Size", border=1, align='C')
        pdf.cell(cell_width, cell_height, txt=f"{config['preprocess_methods'][preprocess_method]['image_size']}", border=1, align='C')
        pdf.ln(cell_height)

        if config['preprocess_method'] == 'tiling':
            pdf.cell(cell_width, cell_height, txt="Stride", border=1, align='C')
            pdf.cell(cell_width, cell_height, txt=f"{config['preprocess_methods']['tiling']['stride']}", border=1, align='C')
            pdf.ln(cell_height)

        learning_model = config['model']
        pdf.cell(cell_width, cell_height, txt="Learning Model", border=1, align='C')
        pdf.cell(cell_width, cell_height, txt=f"{learning_model}", border=1, align='C')
        pdf.ln(cell_height)

        df = pd.read_csv(hyperparams_filename_in_report_path, index_col=False)
        # model_dict = df.to_dict()

        # for key, value in model_dict.items():
        for idx, row in df.iterrows():
            pdf.cell(cell_width, cell_height, txt=f"{learning_model}: {row['parameter']}", border=1, align='C')
            pdf.cell(cell_width, cell_height, txt=f"{row['value']}", border=1, align='C')
            pdf.ln(cell_height)

        pdf.ln(20)
        pdf.output(os.path.join(report_path, 'hyperparams.pdf'))
        display_hyperparams = True


    data_IRPC['median_mAP'] = data_IRPC.groupby('object_name')['mAP'].transform('median')
    sorted_objects = data_IRPC[['object_name', 'median_mAP']]\
        .drop_duplicates().sort_values(by='median_mAP', ascending=False)['object_name']
    data_IRPC = data_IRPC.drop('median_mAP', axis=1)

    if 'report' in config and 'curves_per_graph' in config['report']:
        num_curves_per_graph = config['report']['curves_per_graph']
        if num_curves_per_graph > 5:
            num_curves_per_graph = 5
    else:
        num_curves_per_graph = 5

    curve_color = ['blue', 'green', 'orange', 'red', 'purple']
    assert len(curve_color) >= num_curves_per_graph

    knee_legend_color = 'black'

    graph_array = []
    curve_array = []
    print("got here 1")
    i = 0
    print(ctxt.report_names)
    for object_name in sorted_objects:
        print(object_name)
        if object_name in ctxt.report_names:
            print(object_name, ctxt.report_names)
            if (i % num_curves_per_graph) == 0:
                print(i)
                if i != 0:
                    print(curve_array)
                    graph_array.append(curve_array)
                curve_array = []
            curve_array.append(object_name)
            print(curve_array)
            i += 1
    if len(curve_array) > 0:
        graph_array.append(curve_array)

    num_graphs = len(graph_array)
    print(graph_array)
    for g_i, curve_array in enumerate(graph_array):
        plt.figure(figsize=(10, 4))
        num_curves = len(curve_array)
        for o_i, object_name in enumerate(curve_array):
            # object_data_IRPC = data_IRPC[data_IRPC['object_name'] == object_name].copy() # TODO take this out later
            object_data_IRPC = data_IRPC[((data_IRPC['object_name'] == object_name) # TODO: put this code in later
                                            & (data_IRPC['knee'] != True))].copy()
            object_data_IRPC = object_data_IRPC.sort_values('degradation_factor').reset_index(drop=True)
            num_data_points = object_data_IRPC.shape[0]
            plt.plot(object_data_IRPC['degradation_factor'], object_data_IRPC['mAP'], label=f"{object_name} IRP Curve",
                      color=curve_color[o_i], marker='o', markersize=6, markeredgecolor='black', markerfacecolor=curve_color[o_i])

            if num_data_points > 1:
                num_xticks = min(5, num_data_points)
                skips = num_data_points // num_xticks

                plt.xticks(object_data_IRPC['degradation_factor'][skips::skips],
                           np.round(object_data_IRPC['GSD'][skips::skips], 2))

            for idx, row in object_data_IRPC.iterrows():
                if row['knee'] == 'unknown':
                    object_data_IRPC.at[idx, 'knee'] = False

            knee_points = data_IRPC[((data_IRPC['object_name'] == object_name) # TODO: put this code in later
                                     & (data_IRPC['knee'] == True))].copy()
            if not knee_points.empty:
                knee_degredation_factor = knee_points['degradation_factor']
                knee_map = knee_points['mAP']
                plt.scatter(knee_degredation_factor, knee_map, color=curve_color[o_i], s=100, zorder=5, marker="^")

        legend_elements = [Line2D([0], [0], marker='^', color='w', label='Knees', markerfacecolor=knee_legend_color, markersize=10)]

        plt.title("Image Resolution-Performance (IRP) Curves with Knee Points")
        plt.xlabel("Ground Sample Distance per pixel (meters)")
        plt.ylabel("Mean Average Precision (mAP)")

        plt.legend(# loc='upper left',
                    handles=plt.gca().get_legend_handles_labels()[0] + legend_elements)


        plt.grid(True)
        print("About to save report curves")
        plt.savefig(os.path.join(report_path, f'irp_curves_{g_i}.pdf'))
        plt.close()

        pdf = FPDF()

        pdf.set_fill_color(200, 220, 255)

        pdf.add_page()
        pdf.set_font("Times", size=11) # Times Roman
        pdf.cell(200, 10, txt="IRP Curve Analysis", ln=True, align='C')
        pdf.ln(10)
        txt = (f"The plot represents {num_curves} IRP curve(s). Each curve corresponds to a specific object detection. "
                + "The x-axis shows the degraded resolution factor, by which the original resolution was reduced and then blown "
                + "back up, effectively degrading the image quality. The y-axis shows the mean average aprecision (mAP) of the "
                + "model's performance. "
                + "The 'knee' triangle points indicate a significant inflection point where the mAP performance starts to plateau."
                )
        pdf.multi_cell(0, 5, txt=txt)

        page_units_used = 0
        cell_height = 5
        first_page = True
        for i, object_name in enumerate(curve_array):
            object_data_IRPC = data_IRPC[data_IRPC['object_name'] == object_name].copy()
            num_lines = object_data_IRPC.shape[0] + 9
            page_units_used += (cell_height * num_lines)
            if page_units_used >= 250 or first_page:
                pdf.add_page()
                page_units_used = (cell_height * num_lines)
            object_data_IRPC = object_data_IRPC.sort_values('degradation_factor').reset_index(drop=True)
            txt = f"Selected data for detection of {object_name}"
            pdf.ln(20)
            pdf.cell(200, 10, txt=txt, ln=True, align='C')
            pdf.ln(4)
            for cell_header in object_data_IRPC:
                this_header = header_to_readable[cell_header]
                if this_header != '' and this_header != 'Object Class':
                    if cell_header == 'mAP':
                        cell_width = 15
                    elif cell_header == 'knee':
                        cell_width = 21
                    elif cell_header == 'GSD':
                        cell_width = 24
                    else:
                        cell_width = 2*len(this_header)
                    pdf.cell(cell_width, cell_height, txt=this_header, border=1, align='C', fill=True)
            pdf.ln(cell_height)
            for idx, row in object_data_IRPC.iterrows():
                for cell_header in object_data_IRPC:
                    this_header = header_to_readable[cell_header]
                    if this_header != '' and this_header != 'Object Class':
                        if cell_header == 'knee':
                            this_valuestr = knee_type_to_readable[row[cell_header]]
                            cell_width = 21
                        elif cell_header == 'mAP':
                            this_valuestr = f"{round(row['mAP'], 3):.{3}f}"
                            cell_width = 15
                        elif cell_header == 'GSD':
                            this_valuestr = f"{round(row['GSD'], 4):.{2}f}"
                            cell_width = 24
                        else:
                            this_valuestr = f"{row[cell_header]}"
                            cell_width = 2*len(this_header)
                        pdf.cell(cell_width, cell_height, txt=this_valuestr, border=1, align='C')
                # Move to the next line after each row
                pdf.ln(cell_height)
            txt = "Legend:"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = "    Width, Height: width and height of the image"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = "    Effective Width, Effective Height: width and height the image was degraded to "
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = "                                                             prior to resizing to the original width and height"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = "    mAP: mean average precision of the bounding boxes for the object class"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = "    GSD (meters): the size (sample) of one dimension of a pixel on the ground"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = "    Pixels on Target: the number of pixels that the object occupies"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            txt = f"    Knee: {knee_type_to_readable[True]} if the data point is a knee, {knee_type_to_readable[False]} if not"
            pdf.cell(1, cell_height, txt=txt, ln=True, align='L')
            first_page = False
        print("About to output anlysis")
        pdf.output(os.path.join(report_path, f'irp_analysis_{g_i}.pdf'))

    merger = PdfMerger()
    if obj_class_filename is not None:
        merger.append(os.path.join(report_path, 'obj_class_table.pdf'))
        os.remove(os.path.join(report_path, 'obj_class_table.pdf'))
    if display_hyperparams:
        merger.append(os.path.join(report_path, 'hyperparams.pdf'))
        os.remove(os.path.join(report_path, 'hyperparams.pdf'))
    for i in range(num_graphs):
        merger.append(os.path.join(report_path, f'irp_curves_{i}.pdf'))
        merger.append(os.path.join(report_path, f'irp_analysis_{i}.pdf'))
        os.remove(os.path.join(report_path, f'irp_curves_{i}.pdf'))
        os.remove(os.path.join(report_path, f'irp_analysis_{i}.pdf'))

    report_filename = os.path.join(report_path, ctxt.report_filename)
    merger.write(report_filename)
    merger.close()

    print(f"Report generated: {ctxt.report_filename}")


