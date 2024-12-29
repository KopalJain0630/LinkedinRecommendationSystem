import os
import json
import time
import argparse
import subprocess
import pandas as pd
import os.path as osp

from rich import print as rprint
from rich.progress import track
    
send_req_cmd_fmt = """curl -H "Authorization: Bearer {}" -H "Content-Type: application/json" -d '{}' "https://api.brightdata.com/datasets/v3/trigger?dataset_id=gd_l1viktl72bvl7bjuj0&include_errors=true" """
data_download_cmd_fmt = """curl -H "Authorization: Bearer {}" "https://api.brightdata.com/datasets/v3/snapshot/{}?format={}" >> {} """


def send_request(args):
    api_token = str(input("Please enter your API token: "))
    df = pd.read_csv(args.input)
    df_len = len(df)
    df_shape = df.shape
    df_cols = list(df.columns)
    rprint(f"There are {df_shape[0]} rows and {df_shape[1]} columns.")
    rprint(f"Name of the columns are as follows: {df_cols}")
    count_per_req = max(5, min(50, args.count_per_req))
    start_idx = args.start
    end_idx = args.end

    url_list = []
    s_id = args.start+1

    out_path = osp.join(args.output_dir, 'req.csv')

    result = []
    if osp.exists(out_path):
        old_df = pd.read_csv(out_path).values
        result = old_df.tolist()
    
    for ridx, row in track(df.iterrows(), total=df_len, description='Scrapping Linkedin Profiles: '):
        if ridx<start_idx:
            continue
        if ridx>=end_idx:
            break
        
        if str(row["URL"])!='nan':
            url_list.append({"url": str(row["URL"])})

        if (ridx+1)%count_per_req==0:
            send_req_cmd = send_req_cmd_fmt.format(api_token, json.dumps(url_list))
            if args.debug:
                print(send_req_cmd)
                out = "debugging"
            else:
                out = subprocess.run(send_req_cmd, shell=True, capture_output=True, text=True).stdout
            url_list = []
            result.append([s_id, ridx+1, out])
            s_id = ridx+1

    if len(url_list):
        send_req_cmd = send_req_cmd_fmt.format(api_token, json.dumps(url_list))
        if args.debug:
            print(send_req_cmd)
            out = "debugging"
        else:
            out = subprocess.run(send_req_cmd, shell=True, capture_output=True, text=True).stdout
        result.append([s_id, ridx+1, out])
    
    out_df = pd.DataFrame(result, columns=["start_idx", "end_idx", "Snapshot_ID"])
    out_df.to_csv(out_path, index=False) 

def download_data(args):
    api_token = str(input("Please enter your API token: "))
    df = pd.read_csv(args.input)
    df_len = len(df)
    for ridx, row in track(df.iterrows(), total=df_len, description='Downloading Linkedin Profiles: '):
        fname = osp.join(args.output_dir, "id_{:05f}_{:05f}.{}".format(row['start_idx'], row['end_idx'], args.file_format))
        if osp.exists(fname):
            print("{} already exists!!")
            continue

        if str(row["Snapshot_ID"])!='nan':
            data_download_cmd = data_download_cmd_fmt.format(api_token, row['Snapshot_ID'], args.file_format, fname)
            if args.debug:
                print(data_download_cmd)
            else:
                subprocess.run(data_download_cmd, shell=True)
            print("Successfully downloaded data at {}".format(fname))
    print("Successfully completed!!")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--send_request', action='store_true', help='Send request to bright data')
    parser.add_argument('--download_data', action='store_true', help='Download data to bright data')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to the input file')
    parser.add_argument('-o', '--output_dir', type=str, default="results", help='Path to the output directory')
    parser.add_argument('-fmt', '--file_format', type=str, default="csv", help='Format of output data')
    parser.add_argument('-st', '--start', type=int, default=0, help='Path to the output directory')
    parser.add_argument('-et', '--end', type=int, default=100000, help='Path to the output directory')
    parser.add_argument('-c', '--count_per_req', type=int, default=50, help='Number of profiles per request')
    args = parser.parse_args()
    if args.send_request:
        send_request(args)
    if args.download_data:
        download_data(args)
    