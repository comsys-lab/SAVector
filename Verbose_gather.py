import os
import pandas as pd
import argparse

# 폴더 경로 설정
parser = argparse.ArgumentParser(description='')
parser.add_argument('outdir', help='Set report file to read')
args = parser.parse_args()
outdir = args.outdir
folder_path = './' + outdir  # 폴더 경로를 실제 경로로 변경해야 합니다.

file_to_delete = os.path.join(folder_path, "merged_TPW_data.csv")

if os.path.exists(file_to_delete):
    os.remove(file_to_delete)

# 원하는 컬럼 순서 정의
# desired_rows = [
#     "mobilenetv3_large",
#     "Googlenet",
#     "BERT_base_10_MH",
#     "DenseNet169",
#     "resnet_fwd_mod_for16SA",
#     "BERT_large_64_MH",
#     "resnet152_for16SA",
#     "YOLOv3",
#     "BERT_large_256_MH"
# ]

    # "resnet152_for16SA",

desired_rows = [
    "mobilenetv3_large",
    "DenseNet169",
    "resnet_fwd_mod_for16SA",
    "BERT_base_10_MH",
    "BERT_large_64_MH",
    "ViT_huge_16"
]

# desired_rows = [
#     "resnet_fwd_mod_for16SA"
# ]


columns = ["SA dim", "Runtime", "Compute util", "Pod util","P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)",\
 "MAC Power (mW)", "SRAM Power (mW)", "Offmem Power (mW)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)", "MAC Energy", "SRAM Energy", "DRAM Energy", "Energy", "Offmem"]

# 결과를 저장할 데이터프레임
merged_df = pd.DataFrame(columns=columns)

# 처리한 파일 목록을 저장할 리스트
processed_files = []


for row in desired_rows:
    # 폴더 내 모든 CSV 파일에 대해 반복
    for filename in os.listdir(folder_path):
        if row not in filename:
            continue
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Header 다음 row 추출
            # rep_row = df.iloc[0]
            rep_row=df[["SA dim", "Runtime", "Compute util", "Pod util","P. Throughput (GFLOPS/S)", "E. Throughput (GFLOPS/S)",\
    "MAC Power (mW)", "SRAM Power (mW)", "Offmem Power (mW)", "Power Consumption (mW)", "P.Throughput/Watt (TOPS/W)", "E.Throughput/Watt (TOPS/W)", "MAC Energy", "SRAM Energy", "DRAM Energy", "Energy", "Offmem"]]
            # print(rep_row)
            
            # # 원하는 순서대로 row를 추가
            # for row in desired_rows:
            #     if row in filename:
            #         # merged_df[row] = rep_row
            #         merged_df = pd.concat([merged_df, rep_row], ignore_index=True)
            merged_df = pd.concat([merged_df, rep_row], ignore_index=True)
            
            # 처리한 파일 명 기록
            processed_files.append(filename)

# 결과를 새로운 CSV 파일로 저장
merged_df.to_csv(folder_path+'/merged_TPW_data.csv', index=False)

# 처리한 파일 목록 출력
for i, filename in enumerate(processed_files, 1):
    print(f"{i}. 처리한 파일: {filename}")

print("데이터를 merged_TPW_data.csv 파일로 저장했습니다.")