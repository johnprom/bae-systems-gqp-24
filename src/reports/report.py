from fpdf import FPDF
import math
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import pandas as pd
from PyPDF2 import PdfMerger

from knee_discovery.knee_discovery import calc_degradation_factor

def generate_report(ctxt):
    config = ctxt.config
    output_top_dir = ctxt.get_output_dir_path()
    results_path = os.path.join(output_top_dir, config['knee_discovery']['output_subdir'])
    results_filename = os.path.join(results_path, config['knee_discovery']['eval_results_filename'])

    data_IAPC = pd.read_csv(results_filename)
    
    # Extract columns
    # object_name = data_IAPC['object_name']
    # orig_res_w = data_IAPC['original_resolution_width'].astype(float)
    # orig_res_h = data_IAPC['original_resolution_height'].astype(float)
    # eff_res_w = data_IAPC['effective_resolution_width'].astype(float)
    # eff_res_h = data_IAPC['effective_resolution_height'].astype(float)
    # mAP = data_IAPC['mAP']
    
    # degradation_factor_w = eff_res_w / orig_res_w
    # degradation_factor_h = eff_res_h / orig_res_h
    # degradation_factor_area = degradation_factor_w * degradation_factor_h
    # degradation_factor = degradation_factor_area.apply(math.sqrt)
    data_IAPC['degradation_factor'] = calc_degradation_factor(data_IAPC)
    
    curve_color = ['blue', 'green', 'orange', 'teal']
    knee_color = 'red'
    
    plt.figure(figsize=(10, 6))
    for i, object_name in enumerate(data_IAPC['object_name'].unique()):
        object_data_IAPC = data_IAPC[data_IAPC['object_name'] == object_name]
        
        # x = object_data_IAPC['degradation_factor']
        # y = object_data_IAPC['mAP']
        
        plt.plot(object_data_IAPC['degradation_factor'], object_data_IAPC['mAP'], label=f"{object_name} IAP Curve", 
                 color=curve_color[i])
        
        # Extract knee points for the object
        knee_points = object_data_IAPC[object_data_IAPC['knee'] == True]
        if not knee_points.empty:
            knee_degredation_factor = knee_points['degradation_factor'].values[0]
            knee_map = knee_points['mAP'].values[0]
            plt.scatter(knee_degredation_factor, knee_map, color=knee_color)
            plt.legend(loc='lower right')
            # plt.legend([f"{object_name} (knee degredation at {knee_degredation_factor}, {knee_map:.2f})"])
    
    # for idx, row in knee_points.iterrows():
    #     plt.text(row['degradation_factor'], row['mAP'] + 0.02, 'knee', fontsize=12, color='red')
    
    # plt.title("IAP Curve with Knee Point(s)")
    # plt.xlabel("Effective Resolution (pixels per 300mm)")
    # plt.ylabel("Mean Average Precision (mAP)")
    # plt.legend()
    # plt.savefig('iap_curve.pdf')
    # plt.close()
    
    # plt.title("IAP Curves with Knee Points")
    # plt.xlabel("Effective Resolution (pixels per 300mm)")
    # plt.ylabel("Mean Average Precision (mAP)")
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    # plt.legend(loc='upper left')
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Knees', markerfacecolor=knee_color, markersize=10)]
    
    # Configure plot
    plt.title("IAP Curves with Knee Points")
    plt.xlabel("Degraded Resolution Factor")
    plt.ylabel("Mean Average Precision (mAP)")
    
    # Add the custom legend with curves and knee point
    # plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., 
    #            handles=plt.gca().get_legend_handles_labels()[0] + legend_elements)
    plt.legend(loc='upper left',  
               handles=plt.gca().get_legend_handles_labels()[0] + legend_elements)
    
    
    plt.grid(True)
    plt.savefig('iap_curves.pdf')
    plt.close()
    
    # Step 3: Create a PDF with explanatory text using FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="IAP Curve Analysis", ln=True, align='C')
    pdf.ln(10)
    txt = ("The plot represents the IAP curve. Each curve corresponds to a specific object detection. "
           + "The x-axis shows the degraded resolution factor, by which the original resolution was reduced and then blown back up, "
           + "effectively degrading the image quality. The y-axis shows the mean average aprecision (mAP) of the model's performance. "
           + "The red-marked 'knee' points indicate a significant inflection point where the mAP performance starts to plateau."
           )
    pdf.multi_cell(0, 10, txt=txt)
    pdf.output("iap_analysis.pdf")
    
    # Step 4: Merge the IAP curve and the analysis text into one PDF using PyPDF2
    merger = PdfMerger()
    merger.append('iap_curves.pdf')
    merger.append('iap_analysis.pdf')
    merger.write('full_report.pdf')
    merger.close()
    
    print("Full report generated: full_report.pdf")
    
