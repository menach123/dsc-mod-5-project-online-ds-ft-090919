import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
state_abb ={"Alabama":"AL","Alaska":"AK","Arizona":"AZ","Arkansas":"AR","California":"CA","Colorado":"CO","Connecticut":"CT","Delaware":"DE","Florida":"FL","Georgia":"GA","Hawaii":"HI","Idaho":"ID","Illinois":"IL","Indiana":"IN","Iowa":"IA","Kansas":"KS","Kentucky":"KY","Louisiana":"LA","Maine":"ME","Maryland":"MD","Massachusetts":"MA","Michigan":"MI","Minnesota":"MN","Mississippi":"MS","Missouri":"MO","Montana":"MT","Nebraska":"NE","Nevada":"NV","New Hampshire":"NH","New Jersey":"NJ","New Mexico":"NM","New York":"NY","North Carolina":"NC","North Dakota":"ND","Ohio":"OH","Oklahoma":"OK","Oregon":"OR","Pennsylvania":"PA","Rhode Island":"RI","South Carolina":"SC","South Dakota":"SD","Tennessee":"TN","Texas":"TX","Utah":"UT","Vermont":"VT","Virginia":"VA","Washington":"WA","West Virginia":"WV","Wisconsin":"WI","Wyoming":"WY","Puerto Rico":"PR","District of Columbia":"DC"}

def upload_csv_acs(text):
    return pd.read_csv(text, skiprows=1)

def removed_error(data):
    for i in data.columns:
        if 'Error' in i:
            data = data.drop(columns = [i])
    return data

def upload_county_acs_data(text):
    data = upload_csv_acs(text)

    #Removing error columns
    data = removed_error(data)

            


    #Remove X columns
    for i in data.columns:
        if (data[i].astype(str) == '(X)').sum():
            data = data.drop(columns = [i])   

    data['County'] = data['Geographic Area Name'].apply(lambda x: x.split(',')[0])
    data['County'] = data['County'].apply(lambda x: x.replace(' County',''))
    data['County'] = data['County'].apply(lambda x: x.replace(' city',''))
    
    data['County'] = data['County'].apply(lambda x: x.replace(' Parish',''))
    data['County'] = data['County'].apply(lambda x: x.replace(' Municipality',''))
    data['County'] = data['County'].apply(lambda x: x.replace(' Borough',''))
    data['County'] = data['County'].apply(lambda x: x.replace(' City and',''))
    data['County'] = data['County'].apply(lambda x: x.replace(' Census Area',''))
    data['State'] = data['Geographic Area Name'].apply(lambda x: x.split(',')[1])
    data.State = data.State.apply(lambda x: state_abb[x[1:]])
    data['CountyState'] = data.County + data.State
    data['CountyState'] = data['CountyState'].apply(lambda x: x.lower())


    transfer_dict = {}
    for i in data.columns:
        j = i.replace('!!',' ')
        j = j.replace(' ','_')
        j = j.replace('Estimate_','')
        transfer_dict.update({i:j})

    data = data.rename(columns=transfer_dict)

    #converting 'N' and '-' to zeros
    data = data.apply(lambda x: x.replace('N', 0.0))
    data = data.apply(lambda x: x.replace('-', 0.0))

    for i in data.columns:
        if data[i].dtype == object:
            try:
                data[i] = data[i].astype(float)
            except:
                continue


    return data

def cms_data():
    data = pd.read_csv('Inpatient_Prospective_Payment_System__IPPS__Provider_Summary_for_the_Top_100_Diagnosis-Related_Groups__DRG__-_FY2011.csv')

    data['DRG_id'] = data['DRG Definition'].apply(lambda x: x[:3])

    data['DRG_label'] = data['DRG Definition'].apply(lambda x: x.split(' - ')[1].split(' W')[0])

    data['without_ccmcc'] = data['DRG Definition'].apply(lambda x: 1 if ' W/O ' in x else 0)
    data['with_mcc'] = data['DRG Definition'].apply(lambda x: 1 if ' W MCC' in x else 0)
    data['with_cc'] = data['DRG Definition'].apply(lambda x: 1 if ' W CC'  in x else 0)
    data['with_ccmcc'] = data['DRG Definition'].apply(lambda x: 1 if ' W CC/MCC'  in x else 0)
    data.with_cc = data.with_cc - data.with_ccmcc

    # Calculating Ratio of Average Payment to the maximum payment for the particular procedure
    dict_transfer = {}
    for i in data.DRG_id.unique():
        dict_transfer.update({i:data.loc[data.DRG_id == i][' Average Total Payments '].max()})
    data['max_payment'] = data.DRG_id.apply(lambda x: dict_transfer[x])
    data['ratio_to_max_payment'] = data[' Average Total Payments ']/ data.DRG_id.apply(lambda x: dict_transfer[x])
    for i in data.DRG_id.unique():
        dict_transfer.update({i:data.loc[data.DRG_id == i][' Average Total Payments '].median()})
    data['median_payment'] = data.DRG_id.apply(lambda x: dict_transfer[x])

    # Calculating Ratio of average discharge to the maximum discharge for the particular procedure
    dict_transfer = {}
    for i in data.DRG_id.unique():
        dict_transfer.update({i:data.loc[data.DRG_id == i][' Total Discharges '].max()})
    data['ratio_to_max_discharge'] = data[' Total Discharges ']/ data.DRG_id.apply(lambda x: dict_transfer[x])

    #Creating dummy columns for DRG Labels
    dummy = pd.get_dummies(data.DRG_label)

    for i in dummy.columns:
        data[i] = dummy[i]


    transfer_dict = {}
    for i in data.columns:
        j = i.replace('!!','_')
        j = j.replace(' ', '_')
        transfer_dict.update({i:j})
    data = data.rename(columns = transfer_dict)


    data['CountyState'] = data.county_name + data.Provider_State
    data = data.rename(columns={'county_name':'County', 'Provider_State':'State'})
    data['CountyState'] = data['CountyState'].str.lower()
    data['CountyState'] = data['CountyState'].apply(lambda x: x.replace('saint','st.'))
    return data

def separate_num_columns(df):
    num_df = df
    object_columns = []
    for i in df.columns:
        if df[i].dtype == object:
            num_df = num_df.drop(columns =i)
            object_columns.append(i)
    object_df = df[object_columns]
    return num_df, object_df

def cms_procedure_dummy_labels(df):
    columns = [i.replace(' ','_') for i in df.DRG_label.unique()]
    for i in ['without_ccmcc', 'with_mcc', 'with_cc', 'with_ccmcc']:
        columns.append(i)
    return df[columns]

def merge_acs_data(df, merged_df):
    object_columns = []
    for i in df.columns:
        if df[i].dtype == object:
            object_columns.append(i)
    
    return df.merge(merged_df, on=object_columns)

def remove_duplicate_countystate(df):
    counts = df.CountyState.value_counts()
    counts = counts.loc[counts == 2].index
    index_ = []
    for i in counts:
        i= df.loc[df.CountyState.apply(lambda x: x == i)][['SEX_AND_AGE_Total_population']].idxmin()
        df = df.drop(index=i)
    return df

def remove_cms(df):
    drop_columns =["Provider_Id", "Provider_Zip_Code", "_Total_Discharges_", "_Average_Covered_Charges_", "_Average_Total_Payments_", "Average_Medicare_Payments", "max_payment", "ratio_to_max_payment", "median_payment", "ratio_to_max_discharge"] 
    return df.drop(columns=drop_columns), df[drop_columns]