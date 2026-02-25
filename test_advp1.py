import os
import re
import pandas as pd
import json

# Script for testing, require a directory of resulting table, where each test case name is {pmid}_{pmcid}.csv
# run by pytest test_advp1.py
# Test log is in test_logs with detail of error for each table

# Paper tested
test_papers_info = [
    (30448613, "PMC6331247"), (30979435, "PMC6783343"), (31055733, "PMC6544706"),  (30617256, "PMC6836675"),
    (30820047, "PMC6463297"), (29458411, "PMC5819208"), (29777097, "PMC5959890"), (30651383, "PMC6369905"),
    (31497858, "PMC6736148"), (30930738, "PMC6425305"), (31426376, "PMC6723529"), (29967939, "PMC6280657"),
    (29107063, "PMC5920782"), (29274321, "PMC5938137"), (30413934, "PMC6358498"), (30805717, "PMC7193309"),
    (30636644, "PMC6330399"), (29752348, "PMC5976227"), (28560309, "PMC5440281"), (27899424, "PMC5237405"),
]

def create_test_tables_from_advp():
    advp1 = pd.read_csv("test_tables/advp.variant.records.hg38.tsv", sep = "\t")

    # modify column name of those used to test
    advp1 = advp1.rename({
        "Top SNP": "SNP",
        "RA 1(Reported Allele 1)": "RA",
        "OR_nonref": "Effect",
        "#dbSNP_hg38_chr": "Chr",
        "dbSNP_hg38_position": "Pos",
        "Cohort_simple3": "Cohort",
        "Population_map": "Population"
    }, axis = 1)[[
        "Pubmed PMID", "SNP", "RA", "P-value", "Effect", "Chr", "Pos", "Cohort", "Population"
    ]]

    # Update chr to right format
    advp1["Chr"] = advp1["Chr"].apply(lambda x: (x[3:]))

    for pmid, pmcid in test_papers_info:
        advp1_with_pmid = advp1[advp1["Pubmed PMID"] == pmid]
        # sort by snp
        advp1_with_pmid = advp1_with_pmid.sort_values("SNP").reset_index().drop("index", axis = 1)
        advp1_with_pmid.to_csv(f"test_tables/{pmid}_{pmcid}.csv", index = False)

def import_table_and_test_table(dir_path, file_name):
    if ".csv" in file_name:
        curr_df = pd.read_csv(f"{dir_path}/{file_name}")
        test_df = pd.read_csv(f"test_tables/{file_name[:-4]}.csv")
    elif ".xlsx" in file_name:
        curr_df = pd.read_excel(f"{dir_path}/{file_name}")
        test_df = pd.read_csv(f"test_tables/{file_name[:-5]}.csv")
    return curr_df, test_df

def test_table_dir_exists(dir_path):
    assert dir_path is not None, "Please provide --dir_path"

def test_table_name(dir_path):
    # We require table name to be in the format of pmid_pmcid.csv
    table_name_pattern = r"^\d+_PMC\d+$"
    for file_name in os.listdir(dir_path):
        # extract filename without tag and check the right pattern
        if ".csv" in file_name:
            assert re.search(table_name_pattern, file_name[:-4]), f"Table {file_name} does not have right name (pmid_pmcid)"
        elif ".xlsx" in file_name:
            assert re.search(table_name_pattern, file_name[:-5]), f"Table {file_name} does not have right name (pmid_pmcid)"
        else:
            raise Exception("Error: table must be .csv or .xlsx")
        assert f"{file_name}" in os.listdir("test_tables"), f"Table {file_name} is in a paper not in test set"

# def test_table_format(dir_path):
#     # Test if table is in right format
#     col_lst = ["SNP", "RA", "P-value", "Effect", "Chr", "Pos", "Cohort", "Population"]
#     for file_name in os.listdir(dir_path):
#         if ".csv" in file_name:
#             curr_df = pd.read_csv(f"{dir_path}/{file_name}")
#         elif ".xlsx" in file_name:
#             curr_df = pd.read_excel(f"{dir_path}/{file_name}")
#         for col in col_lst:
#             assert col in curr_df.columns, f"Table {file_name} does not have column {col}"

def test_unique_snp(dir_path):
    # Test if we have the right set of snp
    failed_table = [] # store (table, error)
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        if "SNP" not in curr_df.columns:
            failed_table.append((file_name, f"Table {file_name} does not have SNP column"))
        else:
            curr_unique_snp = set(curr_df["SNP"].unique())
            test_unique_snp = set(test_df["SNP"].unique())
            if curr_unique_snp != test_unique_snp:
                failed_table.append((file_name, f"Table {file_name} do not have the right SNP, differ at: {curr_unique_snp - test_unique_snp} vs {test_unique_snp - curr_unique_snp}"))
    try:
        assert len(failed_table) == 0
    except AssertionError:
        print(f"Failed test_unique_snp on {len(failed_table)}")
        with open("test_logs/test_unique_snp.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise


def test_num_record_snp(dir_path):
    # test if we have the right number of row for each snp
    failed_table = []
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        if "SNP" not in curr_df.columns:
            failed_table.append((file_name, f"Table {file_name} does not have SNP column"))
        else:
            test_unique_snp = test_df["SNP"].unique()
            for snp in test_unique_snp:
                curr_snp_df = curr_df[curr_df["SNP"] == snp]
                test_snp_df = test_df[test_df["SNP"] == snp]
                if curr_snp_df.shape[0] != test_snp_df.shape[0]:
                    failed_table.append((file_name, f"Table {file_name} does not have the right number of row for SNP {snp}"))
                    break
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_num_record_snp.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_num_record_snp on {len(failed_table)} tables")

def get_failed_table_for_test(dir_path, col):
    failed_table = []
    for file_name in os.listdir(dir_path):
        curr_df, test_df = import_table_and_test_table(dir_path, file_name)
        if "SNP" not in curr_df.columns:
            failed_table.append((file_name, f"Table {file_name} does not have SNP column"))
        else:
            if col not in curr_df.columns:
                failed_table.append((file_name, f"Table {file_name} does not have {col} column"))
            else:
                test_unique_snp = test_df["SNP"].unique()
                for snp in test_unique_snp:
                    curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", col]].sort_values(col).reset_index().drop("index", axis = 1)
                    curr_snp_col = curr_snp_df[col]
                    test_snp_df = test_df[test_df["SNP"] == snp][["SNP", col]].sort_values(col).reset_index().drop("index", axis = 1)
                    test_snp_col = test_snp_df[col]
                    try:
                        if not (curr_snp_col == test_snp_col).all():
                            failed_table.append((file_name, f"Table {file_name} does not contain right set of {col} for SNP {snp}"))
                            break
                    except:
                        failed_table.append((file_name, f"Table {file_name} does not contain right set of {col} for SNP {snp}"))
                        break
    return failed_table

def test_snp_ra(dir_path):
    failed_table = get_failed_table_for_test(dir_path, "RA")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_ra.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_ra on {len(failed_table)} tables")

def test_snp_chr(dir_path):
    # test for each table and for each snp we have right set of Chr
    failed_table = get_failed_table_for_test(dir_path, "Chr")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_chr.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_chr on {len(failed_table)} tables")

def test_snp_pos(dir_path):
    # test for each table and for each snp we have right set of Pos
    failed_table = get_failed_table_for_test(dir_path, "Pos")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_pos.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_pos on {len(failed_table)} tables")

def test_snp_effect(dir_path):
    # test for each table and for each snp we have right set of effect
    failed_table = get_failed_table_for_test(dir_path, "Effect")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_effect.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_effect on {len(failed_table)} tables")

# def test_snp_effect_str(dir_path):
#     # test for each table and for each snp we have right set of effect (given in str form)
#     for file_name in os.listdir(dir_path):
#         curr_df, test_df = import_table_and_test_table(dir_path, file_name)
#         test_unique_snp = test_df["SNP"].unique()
#         for snp in test_unique_snp:
#             curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "Effect"]].sort_values("Effect")
#             curr_snp_effect = curr_snp_df["Effect"]
#             test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "Effect"]].sort_values("Effect")
#             test_snp_effect = test_snp_df["Effect"].apply(lambda x: str(x))
#             assert (curr_snp_effect == test_snp_effect).all(), f"Table {file_name} does not contain right set of effect for SNP {snp}"

def test_snp_pvalue(dir_path):
    # test for each table and for each snp we have right set of p-value (numerically)
    failed_table = get_failed_table_for_test(dir_path, "P-value")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_pvalue.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_pvalue on {len(failed_table)} tables")

# def test_snp_pvalue_str(dir_path):
#     # test for each table and for each snp we have right set of p-value (str)
#     for file_name in os.listdir(dir_path):
#         curr_df, test_df = import_table_and_test_table(dir_path, file_name)
#         test_unique_snp = test_df["SNP"].unique()
#         for snp in test_unique_snp:
#             curr_snp_df = curr_df[curr_df["SNP"] == snp][["SNP", "P-value"]].sort_values("P-value")
#             curr_snp_pvalue = curr_snp_df["P-value"]
#             test_snp_df = test_df[test_df["SNP"] == snp][["SNP", "P-value"]].sort_values("P-value")
#             test_snp_pvalue = test_snp_df["P-value"].apply(lambda x: str(x))
#             assert (curr_snp_pvalue == test_snp_pvalue).all(), f"Table {file_name} does not contain right set of p-value for SNP {snp}"

def test_snp_cohort(dir_path):
    # test for each table and for each snp we have right set of cohort
    failed_table = get_failed_table_for_test(dir_path, "Cohort")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_cohort.json", "w") as f:
            json.dump(failed_table, f, indent=2) 
        raise AssertionError(f"Failed test_snp_cohort on {len(failed_table)} tables")

def test_snp_population(dir_path):
    # test for each table and for each snp we have right set of population
    failed_table = get_failed_table_for_test(dir_path, "Population")
    try:
        assert len(failed_table) == 0
    except AssertionError:
        with open("test_logs/test_snp_population.json", "w") as f:
            json.dump(failed_table, f, indent=2)
        raise AssertionError(f"Failed test_snp_population on {len(failed_table)} tables")

# if __name__ == "__main__":
    # create_test_tables_from_advp()
