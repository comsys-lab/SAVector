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
desired_columns = [
    "mobilenetv3_large",
    "Googlenet",
    "BERT_base_10_MH",
    "DenseNet169",
    "resnet_fwd_mod_for16SA",
    "BERT_large_64_MH",
    "resnet152_for16SA",
    "YOLOv3",
    "BERT_large_256_MH"
]

# 결과를 저장할 데이터프레임
merged_df = pd.DataFrame(columns=desired_columns)

# 처리한 파일 목록을 저장할 리스트
processed_files = []

# 폴더 내 모든 CSV 파일에 대해 반복
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # "E.Throughput/Watt (TOPS/W)" 열 추출
        energy_column = df["E.Throughput/Watt (TOPS/W)"]
        
        # 원하는 순서대로 컬럼을 추가
        for column in desired_columns:
            if column in filename:
                merged_df[column] = energy_column
        
        # 처리한 파일 명 기록
        processed_files.append(filename)

# 결과를 새로운 CSV 파일로 저장
merged_df.to_csv(folder_path+'/merged_TPW_data.csv', index=False)

# 처리한 파일 목록 출력
for i, filename in enumerate(processed_files, 1):
    print(f"{i}. 처리한 파일: {filename}")

print("데이터를 merged_TPW_data.csv 파일로 저장했습니다.")