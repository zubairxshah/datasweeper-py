import streamlit as st
import pandas as pd
import os
from io import BytesIO

# Set up the app
st.set_page_config(page_title="💿 Data Sweeper", page_icon=":cdrom:", layout="wide")
st.title("💿 Data Sweeper")
st.write("""Welcome to Data Sweeper! This app transforms your files between CSV and Excel formats with built-in data 
         cleaning and visualization!""")

def find_header_row(df_raw):
    """Find the row index that likely contains the header based on content."""
    for i, row in df_raw.iterrows():
        non_empty = row.notna().sum()
        if non_empty >= 3 and any(col in row.values for col in ['Sr.#', 'Type', 'Description', 'Qty.', 'Rate', 'Amount', 'Project Name']):
            return i
    return 0  # Default to first row if no header found

def clean_dataframe(df):
    """Clean the DataFrame by removing mostly empty rows and forcing numeric types."""
    # Keep rows with at least 3 non-NaN values (adjust as needed)
    df = df[df.notna().sum(axis=1) >= 3]
    # Drop fully empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]
    # Force numeric columns
    for col in ['Qty.', 'Rate', 'Amount', 'Days Required', 'Progress']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

uploaded_files = st.file_uploader("🔺 Upload your files (CSV or Excel):", type=("csv", "xlsx"), accept_multiple_files=True)

if uploaded_files:
    # Store processed DataFrames in a dictionary
    file_data = {}
    
    # Process each uploaded file
    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[-1].lower()
        try:
            if file_ext == ".csv":
                df_raw = pd.read_csv(file, header=None)
                header_row = find_header_row(df_raw)
                file.seek(0)
                df = pd.read_csv(file, skiprows=header_row, header=0)
            elif file_ext == ".xlsx":
                df_raw = pd.read_excel(file, header=None)
                header_row = find_header_row(df_raw)
                file.seek(0)
                df = pd.read_excel(file, skiprows=header_row, header=0)
            else:
                st.error(f"Unsupported file format: {file_ext}")
                continue

            # Clean the DataFrame
            df = clean_dataframe(df)
            file_data[file.name] = df

        except pd.errors.EmptyDataError:
            st.error(f"Error: No data found in {file.name} after skipping rows.")
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")

    if file_data:
        # File selection dropdown
        selected_file = st.selectbox("📰 Select a file to view/process:", list(file_data.keys()))
        df = file_data[selected_file]

        # Display info about the selected file
        st.subheader(f"File: {selected_file}")
        file_size = next(f.size for f in uploaded_files if f.name == selected_file)
        st.subheader(f"File size: {file_size / 1024:.2f} KB")

        # Show 5 rows of the dataframe
        st.write("Preview the head of the Dataframe")
        st.write(df.head())

        # Options for data cleaning
        st.subheader("✂ Data Cleaning Options")
        if st.checkbox(f"Clean data for {selected_file}", key=f"clean_{selected_file}"):
            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"Remove Duplicates from {selected_file}", key=f"dup_{selected_file}"):
                    df.drop_duplicates(inplace=True)
                    file_data[selected_file] = df  # Update stored DataFrame
                    st.write("Duplicates removed!")

            with col2:
                if st.button(f"Fill Missing Values for {selected_file}", key=f"fill_{selected_file}"):
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    file_data[selected_file] = df  # Update stored DataFrame
                    st.write("Missing values have been filled!")

        # Choose specific columns to keep or convert
        st.subheader("📌 Select Columns to Convert")
        columns = st.multiselect(f"Select columns for {selected_file}", df.columns, default=list(df.columns), key=f"cols_{selected_file}")
        df = df[columns]
        file_data[selected_file] = df  # Update stored DataFrame

        # Create some visualization
        st.subheader("📊 Data Visualization")
        if st.checkbox(f"Show Visualization for {selected_file}", key=f"viz_{selected_file}"):
            numeric_df = df.select_dtypes(include='number')
            if not numeric_df.empty and len(numeric_df.columns) >= 1:
                plot_df = numeric_df.iloc[:, :min(2, len(numeric_df.columns))].copy()
                plot_df.columns = [f"col_{i}" for i in range(len(plot_df.columns))]  # Safe column names
                st.bar_chart(plot_df)
            else:
                st.warning(f"No numeric columns available to plot for {selected_file}")

        # Convert the file -> CSV to Excel or vice versa
        st.subheader("🧩 Conversion Options")
        conversion_type = st.radio(f"Convert {selected_file} to:", ["CSV", "Excel"], key=f"conv_{selected_file}")
        if st.button(f"Convert {selected_file}", key=f"convert_{selected_file}"):
            buffer = BytesIO()
            if conversion_type == "CSV":
                df.to_csv(buffer, index=False)
                new_file_name = selected_file.rsplit('.', 1)[0] + ".csv"
                mime_type = "text/csv"
            elif conversion_type == "Excel":
                df.to_excel(buffer, index=False)
                new_file_name = selected_file.rsplit('.', 1)[0] + ".xlsx"
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            buffer.seek(0)

            # Download Button
            st.download_button(
                label=f"🔻 Download {selected_file} as {conversion_type}",
                data=buffer,
                file_name=new_file_name,
                mime=mime_type,
                key=f"download_{selected_file}"
            )

        st.success("Thank you for using Data Sweeper! 🚀")
    else:
        st.warning("No valid files processed. Check your uploads.")
else:
    st.info("Please add files to get started")