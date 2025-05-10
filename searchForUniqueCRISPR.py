#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

def read_sequences(folder_path):
    """
    Функция для чтения файлов из папки CRISPRS.
    Ищет файлы с расширениями .fa, .fasta, .fna, считывает их с помощью Biopython
    и формирует DataFrame, где каждая колонка соответствует файлу, а строки – спейсеры.
    """
    seq_dict = {}
    allowed_extensions = ('*.fa', '*.fasta', '*.fna')
    for ext in allowed_extensions:
        files = glob.glob(os.path.join(folder_path, ext))
        for file in files:
            base_name = os.path.basename(file)
            sequences = []
            # Используем SeqIO для парсинга файла
            for record in SeqIO.parse(file, "fasta"):
                sequences.append(str(record.seq))
            if sequences:  # Только если в файле есть данные
                seq_dict[base_name] = sequences
    # Формируем DataFrame – если в одном файле меньше спейсеров, недостающие строки получат значение NaN
    df = pd.DataFrame({k: pd.Series(v) for k, v in seq_dict.items()})
    return df

def pad_sequences(df):
    """
    Функция принимает DataFrame с последовательностями.
    Для каждой колонки определяется максимальная длина спейсера и все последовательности дополняются символами 'N'
    до этой длины. Результат записывается в новый DataFrame с названием колонок, начинающихся с префикса "P_".
    """
    padded_df = pd.DataFrame()
    for col in df.columns:
        # Убираем возможные NaN и получаем список последовательностей
        col_data = df[col].dropna().tolist()
        if not col_data:
            continue
        max_len = max(len(seq) for seq in col_data)
        padded_sequences = []
        for seq in df[col]:
            if pd.isna(seq):
                padded_sequences.append(seq)
            else:
                # Метод ljust дополняет строку справа символами 'N' до нужной длины
                padded_sequences.append(seq.ljust(max_len, 'N'))
        padded_df["P_" + col] = padded_sequences
    return padded_df

def calculate_similarity(seq_list):
    """
    Функция принимает список последовательностей (уже выровненных с помощью pad_sequences).
    Для каждой пары последовательностей вычисляет процент совпадения по следующим правилам:
       - Расстояние Хэмминга: сравниваются символы на одинаковых позициях.
       - Процент совпадения: (число совпадающих символов / длина последовательности) * 100.
    Результат возвращается в виде DataFrame-матрицы, где и строки, и колонки именуются как spacer1, spacer2, ...
    """
    n = len(seq_list)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 100.0
            else:
                s1 = seq_list[i]
                s2 = seq_list[j]
                matches = sum(1 for a, b in zip(s1, s2) if a == b)
                similarity = (matches / len(s1)) * 100
                sim_matrix[i, j] = similarity
    spacer_labels = [f"spacer{i + 1}" for i in range(n)]
    df_sim = pd.DataFrame(sim_matrix, index=spacer_labels, columns=spacer_labels)
    return df_sim

def export_and_plot(sim_df, output_prefix):
    """
    Функция принимает DataFrame с матрицей сходства, затем:
       - Экспортирует её в CSV-файл с именем <output_prefix>_similarity.csv.
       - Создает тепловую карту (heatmap) с использованием seaborn, где большее совпадение отображается ярче.
         График сохраняется в файл <output_prefix>_heatmap.png.
    """
    csv_filename = f"{output_prefix}_similarity.csv"
    sim_df.to_csv(csv_filename, index=True)
    print(f"Сохранена матрица сходства: {csv_filename}")

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(sim_df, annot=True, cmap="viridis", vmin=0, vmax=100, fmt=".1f")
    plt.title(f"Spacer Similarity Heatmap - {output_prefix}")
    plt.tight_layout()
    heatmap_filename = f"{output_prefix}_heatmap.png"
    plt.savefig(heatmap_filename)
    plt.close()
    print(f"Сохранена тепловая карта: {heatmap_filename}")

def main():
    """
    Main-блок:
      1. Чтение файлов из папки CRISPRS.
      2. Дополнение последовательностей (padding) до максимальной длины в каждом столбце.
      3. Для каждого столбца (файла) по отдельности:
            - Вычисление матрицы сходства для спейсеров.
            - Экспорт результата в CSV и визуализация в виде тепловой карты.
      4. Сбор всех последовательностей воедино для общего анализа,
         вычисление общей матрицы сходства, экспорт в CSV и построение тепловой карты.
    """
    # Определяем путь к папке, в которой находится код, и затем к папке "CRISPRS"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    crisprs_folder = os.path.join(current_dir, "CRISPRS")
    if not os.path.exists(crisprs_folder):
        print(f"Папка {crisprs_folder} не найдена!")
        return
    print("Чтение файлов из папки:", crisprs_folder)

    # Функция 1: Чтение файлов
    df = read_sequences(crisprs_folder)
    print("Исходный DataFrame с последовательностями:")
    print(df)

    # Функция 2: Выравнивание (padding) последовательностей в каждом столбце
    padded_df = pad_sequences(df)
    print("DataFrame с дополненными (padded) последовательностями:")
    print(padded_df)

    overall_sequences = []
    # Обработка каждого файла (каждого столбца) отдельно
    for col in padded_df.columns:
        seq_list = padded_df[col].dropna().tolist()
        if not seq_list:
            continue
        print(f"\nОбработка файла: {col}")
        # Функция 3: Расчет матрицы сходства для списка спейсеров одного файла
        sim_df = calculate_similarity(seq_list)
        # Функция 4: Экспорт матрицы в CSV и построение тепловой карты
        export_and_plot(sim_df, output_prefix=col)
        # Собираем все последовательности для общего анализа
        overall_sequences.extend(seq_list)

    # Общий анализ по всем файлам
    if overall_sequences:
        print("\nОбщий анализ для всех последовательностей:")
        overall_sim_df = calculate_similarity(overall_sequences)
        export_and_plot(overall_sim_df, output_prefix="overall")
    else:
        print("Нет последовательностей для общего анализа.")
if __name__ == '__main__':
    main()