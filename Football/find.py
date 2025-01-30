import csv
from datetime import datetime
import pandas as pd
import os


'''
def searchCSV(source_fname, conditions, result_fname = 'scripts/query.csv'):
    result = []
    try:
        with open(source_fname, mode='r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            with open(result_fname, mode="w", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(header)
                for row in csv_reader:
                    flag = True
                    for col, val in conditions:
                        if not isinstance(val, str):
                            val = str(val)
                        # NOT
                        if val.startswith('!'):
                            val = val[1:]
                            if str(row[header.index(col)])==val:
                                flag = False
                                break
                        # GREATER OR EQUAL
                        elif val.startswith('>='):
                            val = val[2:]
                            try:
                                # Time
                                val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                row_time = datetime.fromisoformat(row[header.index(col)].replace("Z", "+00:00"))
                                if row_time < val_time:
                                    flag = False
                                    break
                            except ValueError:
                                # Number
                                val_num = float(val)
                                row_num = float(row[header.index(col)])
                                if row_num < val_num:
                                    flag = False
                                    break
                            except Exception as e1:
                                print(e1)
                        # GREATER
                        elif val.startswith('>'):
                            val = val[1:]
                            try:
                                # Time
                                val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                row_time = datetime.fromisoformat(row[header.index(col)].replace("Z", "+00:00"))
                                if row_time <= val_time:
                                    flag = False
                                    break
                            except ValueError:
                                # Number
                                val_num = float(val)
                                row_num = float(row[header.index(col)])
                                if row_num <= val_num:
                                    flag = False
                                    break
                            except Exception as e1:
                                print(e1)
                        # LESS OR EQUAL
                        elif val.startswith('<='):
                            val = val[2:]
                            try:
                                # Time
                                val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                row_time = datetime.fromisoformat(row[header.index(col)].replace("Z", "+00:00"))
                                if row_time > val_time:
                                    flag = False
                                    break
                            except ValueError:
                                # Number
                                val_num = float(val)
                                row_num = float(row[header.index(col)])
                                if row_num > val_num:
                                    flag = False
                                    break
                            except Exception as e1:
                                print(e1)
                        # LESS
                        elif val.startswith('<'):
                            val = val[1:]
                            try:
                                # Time
                                val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                                row_time = datetime.fromisoformat(row[header.index(col)].replace("Z", "+00:00"))
                                if row_time >= val_time:
                                    flag = False
                                    break
                            except ValueError:
                                # Number
                                val_num = float(val)
                                row_num = float(row[header.index(col)])
                                if row_num >= val_num:
                                    flag = False
                                    break
                            except Exception as e1:
                                print(e1)
                        # CONTAINS
                        elif val.startswith('.'):
                            val = val[1:]
                            if val not in row[header.index(col)]:
                                flag = False
                                break
                        # EQUAL
                        else:
                            if row[header.index(col)]!=val:
                                flag = False
                                break
                    if flag:
                        csv_writer.writerow(row)
                        result.append(row)
        result_df = pd.DataFrame(result, columns=header)
        return result_df
    except Exception as e:
        print(e)
        
def searchDF(source_df, conditions):
    result_df = pd.DataFrame(columns=source_df.columns)
    
    for index, row in source_df.iterrows():
        flag = True
        for col, val in conditions:
            if not isinstance(val, str):
                val = str(val)
            # NOT
            if val.startswith('!'):
                val = val[1:]
                if str(row[col]) == val:
                    flag = False
                    break
            # GREATER OR EQUAL
            elif val.startswith('>='):
                val = val[2:]
                try:
                    # Time
                    val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    row_time = datetime.fromisoformat(row[col].replace("Z", "+00:00"))
                    if row_time < val_time:
                        flag = False
                        break
                except ValueError:
                    # Number
                    val_num = float(val)
                    row_num = float(row[col])
                    if row_num < val_num:
                        flag = False
                        break
                except Exception as e1:
                    print(e1)
            # GREATER
            elif val.startswith('>'):
                val = val[1:]
                try:
                    # Time
                    val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    row_time = datetime.fromisoformat(row[col].replace("Z", "+00:00"))
                    if row_time <= val_time:
                        flag = False
                        break
                except ValueError:
                    # Number
                    val_num = float(val)
                    row_num = float(row[col])
                    if row_num <= val_num:
                        flag = False
                        break
                except Exception as e1:
                    print(e1)
            # LESS OR EQUAL
            elif val.startswith('<='):
                val = val[2:]
                try:
                    # Time
                    val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    row_time = datetime.fromisoformat(row[col].replace("Z", "+00:00"))
                    if row_time > val_time:
                        flag = False
                        break
                except ValueError:
                    # Number
                    val_num = float(val)
                    row_num = float(row[col])
                    if row_num > val_num:
                        flag = False
                        break
                except Exception as e1:
                    print(e1)
            # LESS
            elif val.startswith('<'):
                val = val[1:]
                try:
                    # Time
                    val_time = datetime.fromisoformat(val.replace("Z", "+00:00"))
                    row_time = datetime.fromisoformat(row[col].replace("Z", "+00:00"))
                    if row_time >= val_time:
                        flag = False
                        break
                except ValueError:
                    # Number
                    val_num = float(val)
                    row_num = float(row[col])
                    if row_num >= val_num:
                        flag = False
                        break
                except Exception as e1:
                    print(e1)
            # CONTAINS
            elif val.startswith('.'):
                val = val[1:]
                if val not in row[col]:
                    flag = False
                    break
            # EQUAL
            else:
                if str(row[col]) != val:
                    flag = False
                    break
        
        if flag:
            result_df = pd.concat([result_df, row.to_frame().T], ignore_index=True)
    return result_df
'''

def apply_conditions(df, conditions):
    for col, val in conditions:
        if isinstance(val, str):
            if val.startswith('!'):
                df = df[df[col] != val[1:]]
            elif val.startswith('>='):
                df = df[pd.to_datetime(df[col], errors='coerce') >= pd.to_datetime(val[2:], errors='coerce')]
            elif val.startswith('>'):
                df = df[pd.to_datetime(df[col], errors='coerce') > pd.to_datetime(val[1:], errors='coerce')]
            elif val.startswith('<='):
                df = df[pd.to_datetime(df[col], errors='coerce') <= pd.to_datetime(val[2:], errors='coerce')]
            elif val.startswith('<'):
                df = df[pd.to_datetime(df[col], errors='coerce') < pd.to_datetime(val[1:], errors='coerce')]
            elif val.startswith('.'):
                df = df[df[col].astype(str).str.contains(val[1:], na=False)]
            else:
                df = df[df[col] == val]
        else:
            df = df[df[col] == val]
    return df

def searchCSV(source_fname, conditions, result_fname='query.csv'):
    try:
        df = pd.read_csv(source_fname)
        result_df = apply_conditions(df, conditions)
        result_df.to_csv(result_fname, index=False)
        return result_df
    except Exception as e:
        print(e)

def searchDF(source_df, conditions):
    return apply_conditions(source_df, conditions)

# searchCSV('Football/match_event.csv', [('match_id', 364)], 'Football/query.csv')