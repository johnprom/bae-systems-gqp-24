#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:43:59 2024

@author: dfox
"""
import sys
from PyPDF2 import PdfReader, PdfWriter

def concatenate_pdfs(output_file, input_files):
    """
    Concatenate multiple PDF files into a single output file.

    Parameters:
        output_file (str): The path to the destination PDF file.
        input_files (list): List of paths to input PDF files.

    Raises:
        Exception: If any input file is not a valid PDF file.
    """
    pdf_writer = PdfWriter()

    for file_path in input_files:
        try:
            pdf_reader = PdfReader(file_path)
        except Exception as e:
            raise Exception(f"Error reading '{file_path}': Not a valid PDF file. Original error: {e}")

        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    with open(output_file, "wb") as output_pdf:
        pdf_writer.write(output_pdf)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 ./append_pdf_files.py <destination file name> <first file name> <second file name> ... <last file name>")
        sys.exit(1)

    destination_file = sys.argv[1]
    source_files = sys.argv[2:]

    try:
        concatenate_pdfs(destination_file, source_files)
        print(f"Successfully created '{destination_file}' by concatenating {len(source_files)} PDF(s).")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
