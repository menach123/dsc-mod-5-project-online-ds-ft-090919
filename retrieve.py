import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def merged_data():
    data = pd.read_csv('Inpatient_Prospective_Payment_System__IPPS__Provider_Summary_for_the_Top_100_Diagnosis-Related_Groups__DRG__-_FY2011.csv')

    data['DRG_id'] = data['DRG Definition'].apply(lambda x: x[:3])

    data['DRG_label'] = data['DRG Definition'].apply(lambda x: x.split(' - ')[1].split(' W')[0])

    data['without_ccmcc'] = data['DRG Definition'].apply(lambda x: 1 if ' W/O ' in x else 0)
    data['with_mcc'] = data['DRG Definition'].apply(lambda x: 1 if ' W MCC' in x else 0)
    data['with_cc'] = data['DRG Definition'].apply(lambda x: 1 if ' W CC'  in x else 0)
    data['with_ccmcc'] = data['DRG Definition'].apply(lambda x: 1 if ' W CC/MCC'  in x else 0)
    data.with_cc = data.with_cc - data.with_ccmcc

    dict_transfer = {}
    for i in data.DRG_id.unique():
        dict_transfer.update({i:data.loc[data.DRG_id == i][' Average Total Payments '].max()})


    data['ratio_to_max_payment'] = data[' Average Total Payments ']/ data.DRG_id.apply(lambda x: dict_transfer[x])

    dict_transfer = {}
    for i in data.DRG_id.unique():
        dict_transfer.update({i:data.loc[data.DRG_id == i][' Total Discharges '].max()})


    data['ratio_to_max_discharge'] = data[' Total Discharges ']/ data.DRG_id.apply(lambda x: dict_transfer[x])

    transfer_dict = {}
    for i in data.columns:
        j = i.replace('!!','_')
        j = j.replace(' ', '_')
        transfer_dict.update({i:j})
    data = data.rename(columns = transfer_dict)


    data['CountyState'] = data.county_name + data.Provider_State
    data['CountyState'] = data['CountyState'].str.lower()
    data['CountyState'] = data['CountyState'].apply(lambda x: x.replace('saint','st.'))

    ### American Community Study

    acs_data = pd.read_csv('ACSDP5Y2017.DP05_data_with_overlays_2019-12-16T161924.csv',skiprows = 1)
    acs_income_data = pd.read_csv('ACSST5Y2017.S1701_data_with_overlays_2019-12-17T171021.csv', skiprows = 1)
    acs_income2_data = pd.read_csv('ACSST1Y2012.S1901_data_with_overlays_2019-12-21T221020.csv', skiprows=1)

    for i in acs_data.columns:
        if 'Error' in i:
            acs_data = acs_data.drop(columns = [i])

    for i in acs_income_data.columns:
        if 'Error' in i:
            acs_income_data = acs_income_data.drop(columns = [i])
            
    for i in acs_income2_data.columns:
        if 'Error' in i:
            acs_income2_data = acs_income2_data.drop(columns = [i])
            
    for i in acs_income2_data.columns:
        if (acs_income2_data[i].astype(str) == '(X)').sum():
            acs_income2_data = acs_income2_data.drop(columns = [i])    

    acs_data['County'] = acs_data['Geographic Area Name'].apply(lambda x: x.split(',')[0][:-7])
    acs_data['State'] = acs_data['Geographic Area Name'].apply(lambda x: x.split(',')[1])

    state_abb ={"Alabama":"AL",
    "Alaska":"AK",
    "Arizona":"AZ",
    "Arkansas":"AR",
    "California":"CA",
    "Colorado":"CO",
    "Connecticut":"CT",
    "Delaware":"DE",
    "Florida":"FL",
    "Georgia":"GA",
    "Hawaii":"HI",
    "Idaho":"ID",
    "Illinois":"IL",
    "Indiana":"IN",
    "Iowa":"IA",
    "Kansas":"KS",
    "Kentucky":"KY",
    "Louisiana":"LA",
    "Maine":"ME",
    "Maryland":"MD",
    "Massachusetts":"MA",
    "Michigan":"MI",
    "Minnesota":"MN",
    "Mississippi":"MS",
    "Missouri":"MO",
    "Montana":"MT",
    "Nebraska":"NE",
    "Nevada":"NV",
    "New Hampshire":"NH",
    "New Jersey":"NJ",
    "New Mexico":"NM",
    "New York":"NY",
    "North Carolina":"NC",
    "North Dakota":"ND",
    "Ohio":"OH",
    "Oklahoma":"OK",
    "Oregon":"OR",
    "Pennsylvania":"PA",
    "Rhode Island":"RI",
    "South Carolina":"SC",
    "South Dakota":"SD",
    "Tennessee":"TN",
    "Texas":"TX",
    "Utah":"UT",
    "Vermont":"VT",
    "Virginia":"VA",
    "Washington":"WA",
    "West Virginia":"WV",
    "Wisconsin":"WI",
    "Wyoming":"WY",
    "Puerto Rico":"PR",
            "District of Columbia":"DC"}



    acs_data.State = acs_data.State.apply(lambda x: state_abb[x[1:]])

    acs_data['CountyState'] = acs_data.County + acs_data.State
    acs_data['CountyState'] = acs_data['CountyState'].str.lower()


    acs_income_data = acs_income_data.drop(columns='Geographic Area Name')
    acs_income2_data = acs_income2_data.drop(columns='Geographic Area Name')
    acs_income_data = acs_income_data.merge(acs_income2_data, how='left', on= 'id')
    acs_data = acs_data.merge(acs_income_data, how='left', on= 'id')

    transfer_dict = {}
    for i in acs_data.columns:
        j = i.replace('!!',' ')
        j = j.replace(' ','_')
        j = j.replace('Estimate_','')
        transfer_dict.update({i:j})

    acs_data = acs_data.rename(columns=transfer_dict)

    hold_df = acs_data[['County','State','CountyState']]

    for i in acs_data.columns:
        if acs_data[i].dtype == object:
            acs_data = acs_data.drop(columns=i)
            
    acs_data[hold_df.columns] = hold_df

    acs_data = acs_data.fillna(0)

    for i in acs_data.columns:
        if acs_data[i].isna().sum() > 0:
            print(i, acs_data[i].isna().sum())

    ### Merging 

    df = acs_data.merge(data, how='right', on='CountyState')

    drop_columns = ['SEX_AND_AGE_Total_population_Male',
                    'Percent_SEX_AND_AGE_Total_population',
    'SEX_AND_AGE_Total_population_Female',
    'SEX_AND_AGE_Total_population_Under_5_years',
    'SEX_AND_AGE_Total_population_5_to_9_years',
    'SEX_AND_AGE_Total_population_10_to_14_years',
    'SEX_AND_AGE_Total_population_15_to_19_years',
    'SEX_AND_AGE_Total_population_20_to_24_years',
    'SEX_AND_AGE_Total_population_25_to_34_years',
    'SEX_AND_AGE_Total_population_35_to_44_years',
    'SEX_AND_AGE_Total_population_45_to_54_years',
    'SEX_AND_AGE_Total_population_55_to_59_years',
    'SEX_AND_AGE_Total_population_60_to_64_years',
    'SEX_AND_AGE_Total_population_65_to_74_years',
    'SEX_AND_AGE_Total_population_75_to_84_years',
    'SEX_AND_AGE_Total_population_85_years_and_over',
    'SEX_AND_AGE_Total_population_Under_18_years',
    'SEX_AND_AGE_Total_population_16_years_and_over',
    'SEX_AND_AGE_Total_population_18_years_and_over',
    'SEX_AND_AGE_Total_population_21_years_and_over',
    'SEX_AND_AGE_Total_population_62_years_and_over',
    'SEX_AND_AGE_Total_population_65_years_and_over',
    'SEX_AND_AGE_Total_population_18_years_and_over.1',
    'SEX_AND_AGE_Total_population_18_years_and_over_Male',
    'SEX_AND_AGE_Total_population_18_years_and_over_Female',
    'SEX_AND_AGE_Total_population_65_years_and_over.1',
    'SEX_AND_AGE_Total_population_65_years_and_over_Male',
    'SEX_AND_AGE_Total_population_65_years_and_over_Female']
    df = df.drop(columns=drop_columns)

    drop_columns = ['RACE_Total_population',
    'RACE_Total_population_One_race',
    'RACE_Total_population_Two_or_more_races',
    'RACE_Total_population_One_race.1',
    'RACE_Total_population_One_race_White',
    'RACE_Total_population_One_race_Black_or_African_American',
    'RACE_Total_population_One_race_American_Indian_and_Alaska_Native',
    'RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Cherokee_tribal_grouping',
    'RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Chippewa_tribal_grouping',
    'RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Navajo_tribal_grouping',
    'RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Sioux_tribal_grouping',
    'RACE_Total_population_One_race_Asian',
    'RACE_Total_population_One_race_Asian_Asian_Indian',
    'RACE_Total_population_One_race_Asian_Chinese',
    'RACE_Total_population_One_race_Asian_Filipino',
    'RACE_Total_population_One_race_Asian_Japanese',
    'RACE_Total_population_One_race_Asian_Korean',
    'RACE_Total_population_One_race_Asian_Vietnamese',
    'RACE_Total_population_One_race_Asian_Other_Asian',
    'RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander',
    'RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Native_Hawaiian',
    'RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Guamanian_or_Chamorro',
    'RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Samoan',
    'RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Other_Pacific_Islander',
    'RACE_Total_population_One_race_Some_other_race',
    'RACE_Total_population_Two_or_more_races.1',
    'RACE_Total_population_Two_or_more_races_White_and_Black_or_African_American',
    'RACE_Total_population_Two_or_more_races_White_and_American_Indian_and_Alaska_Native',
    'RACE_Total_population_Two_or_more_races_White_and_Asian',
    'RACE_Total_population_Two_or_more_races_Black_or_African_American_and_American_Indian_and_Alaska_Native']
    df = df.drop(columns=drop_columns)

    drop_columns = ['Percent_RACE_Total_population',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_White',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Black_or_African_American',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_American_Indian_and_Alaska_Native',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Asian',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Native_Hawaiian_and_Other_Pacific_Islander',
    'Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Some_other_race']
    df = df.drop(columns=drop_columns)

    drop_columns = ['HISPANIC_OR_LATINO_AND_RACE_Total_population',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Mexican',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Puerto_Rican',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Cuban',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Other_Hispanic_or_Latino',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_White_alone',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Black_or_African_American_alone',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_American_Indian_and_Alaska_Native_alone',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Asian_alone',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Native_Hawaiian_and_Other_Pacific_Islander_alone',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Some_other_race_alone',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Two_or_more_races',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Two_or_more_races_Two_races_including_Some_other_race',
    'HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Two_or_more_races_Two_races_excluding_Some_other_race_and_Three_or_more_races']
    df = df.drop(columns=drop_columns)


    drop_columns= ['Below_poverty_level_Population_for_whom_poverty_status_is_determined',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years_Under_5_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years_5_to_17_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years_Related_children_of_householder_under_18_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years_18_to_34_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years_35_to_64_years',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_60_years_and_over',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_65_years_and_over',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_SEX_Male',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_SEX_Female',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_White_alone',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Black_or_African_American_alone',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_American_Indian_and_Alaska_Native_alone',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Asian_alone',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Native_Hawaiian_and_Other_Pacific_Islander_alone',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Some_other_race_alone',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Two_or_more_races',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Hispanic_or_Latino_origin_(of_any_race)',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_White_alone,_not_Hispanic_or_Latino',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Less_than_high_school_graduate',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_High_school_graduate_(includes_equivalency)',
    "Below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Some_college,_associate's_degree",
    "Below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Bachelor's_degree_or_higher",
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed_Male',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed_Female',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Unemployed',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Unemployed_Male',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Unemployed_Female',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Worked_full-time,_year-round_in_the_past_12_months',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Worked_part-time_or_part-year_in_the_past_12_months',
    'Below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Did_not_work',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Male',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Female',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_15_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_16_to_17_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_18_to_24_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_25_to_34_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_35_to_44_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_45_to_54_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_55_to_64_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_65_to_74_years',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_75_years_and_over',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Worked_full-time,_year-round_in_the_past_12_months',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Worked_less_than_full-time,_year-round_in_the_past_12_months',
    'Below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Did_not_work',
                'Unnamed:_0',
    "Total_Population_for_whom_poverty_status_is_determined",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years_Under_5_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years_5_to_17_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_Under_18_years_Related_children_of_householder_under_18_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years_18_to_34_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years_35_to_64_years",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_60_years_and_over",
    "Total_Population_for_whom_poverty_status_is_determined_AGE_65_years_and_over",
    "Total_Population_for_whom_poverty_status_is_determined_SEX_Male",
    "Total_Population_for_whom_poverty_status_is_determined_SEX_Female",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_White_alone",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Black_or_African_American_alone",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_American_Indian_and_Alaska_Native_alone",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Asian_alone",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Native_Hawaiian_and_Other_Pacific_Islander_alone",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Some_other_race_alone",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Two_or_more_races",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_Hispanic_or_Latino_origin_(of_any_race)",
    "Total_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_White_alone,_not_Hispanic_or_Latino",
    "Total_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over",
    "Total_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Less_than_high_school_graduate",
    "Total_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_High_school_graduate_(includes_equivalency)",
    "Total_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Some_college,_associate's_degree",
    "Total_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Bachelor's_degree_or_higher",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed_Male",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed_Female",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Unemployed",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Unemployed_Male",
    "Total_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Unemployed_Female",
    "Total_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over",
    "Total_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Worked_full-time,_year-round_in_the_past_12_months",
    "Total_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Worked_part-time_or_part-year_in_the_past_12_months",
    "Total_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Did_not_work",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_50_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_125_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_150_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_185_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_200_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_300_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_400_percent_of_poverty_level",
    "Total_Population_for_whom_poverty_status_is_determined_ALL_INDIVIDUALS_WITH_INCOME_BELOW_THE_FOLLOWING_POVERTY_RATIOS_500_percent_of_poverty_level",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Male",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Female",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_15_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_16_to_17_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_18_to_24_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_25_to_34_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_35_to_44_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_45_to_54_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_55_to_64_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_65_to_74_years",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_75_years_and_over",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Worked_full-time,_year-round_in_the_past_12_months",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Worked_less_than_full-time,_year-round_in_the_past_12_months",
    "Total_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Did_not_work"]
    df = df.drop(columns=drop_columns)
    dummy = pd.get_dummies(df.DRG_label)
    df[dummy.columns] = dummy
    df = df.drop(columns=['DRG_Definition', 'DRG_label'])
    return df

def clean_data(df):
     
    
    for i in df.columns:
        if df[i].dtype == object:
            df = df.drop(columns=i)
        else:
            df[i] = df[i].astype(float)
    return df

def sex_age_data(df):
     
    select = ['SEX_AND_AGE_Total_population',
    'Percent_SEX_AND_AGE_Total_population_Male',
    'Percent_SEX_AND_AGE_Total_population_Female',
    'SEX_AND_AGE_Total_population_Sex_ratio_(males_per_100_females)',
    'Percent_SEX_AND_AGE_Total_population_Under_5_years',
    'Percent_SEX_AND_AGE_Total_population_5_to_9_years',
    'Percent_SEX_AND_AGE_Total_population_10_to_14_years',
    'Percent_SEX_AND_AGE_Total_population_15_to_19_years',
    'Percent_SEX_AND_AGE_Total_population_20_to_24_years',
    'Percent_SEX_AND_AGE_Total_population_25_to_34_years',
    'Percent_SEX_AND_AGE_Total_population_35_to_44_years',
    'Percent_SEX_AND_AGE_Total_population_45_to_54_years',
    'Percent_SEX_AND_AGE_Total_population_55_to_59_years',
    'Percent_SEX_AND_AGE_Total_population_60_to_64_years',
    'Percent_SEX_AND_AGE_Total_population_65_to_74_years',
    'Percent_SEX_AND_AGE_Total_population_75_to_84_years',
    'Percent_SEX_AND_AGE_Total_population_85_years_and_over',
    'SEX_AND_AGE_Total_population_Median_age_(years)',
    'Percent_SEX_AND_AGE_Total_population_Under_18_years',
    'Percent_SEX_AND_AGE_Total_population_16_years_and_over',
    'Percent_SEX_AND_AGE_Total_population_18_years_and_over',
    'Percent_SEX_AND_AGE_Total_population_21_years_and_over',
    'Percent_SEX_AND_AGE_Total_population_62_years_and_over',
    'Percent_SEX_AND_AGE_Total_population_65_years_and_over',
    'Percent_SEX_AND_AGE_Total_population_18_years_and_over.1',
    'Percent_SEX_AND_AGE_Total_population_18_years_and_over_Male',
    'Percent_SEX_AND_AGE_Total_population_18_years_and_over_Female',
    'SEX_AND_AGE_Total_population_18_years_and_over_Sex_ratio_(males_per_100_females)',
    'Percent_SEX_AND_AGE_Total_population_65_years_and_over.1',
    'Percent_SEX_AND_AGE_Total_population_65_years_and_over_Male',
    'Percent_SEX_AND_AGE_Total_population_65_years_and_over_Female',
    "Percent_CITIZEN_VOTING_AGE_POPULATION_Citizen_18_and_over_population",
    "Percent_CITIZEN_VOTING_AGE_POPULATION_Citizen_18_and_over_population_Male",
    "Percent_CITIZEN_VOTING_AGE_POPULATION_Citizen_18_and_over_population_Female",
    'ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE',
    'ALCOHOL/DRUG ABUSE OR DEPENDENCE',
    'ATHEROSCLEROSIS',
    'BACK & NECK PROC EXC SPINAL FUSION',
    'BRONCHITIS & ASTHMA',
    'CARDIAC ARRHYTHMIA & CONDUCTION DISORDERS',
    'CELLULITIS',
    'CERVICAL SPINAL FUSION',
    'CHEST PAIN',
    'CHRONIC OBSTRUCTIVE PULMONARY DISEASE',
    'CIRCULATORY DISORDERS EXCEPT AMI,',
    'CRANIAL & PERIPHERAL NERVE DISORDERS',
    'DEGENERATIVE NERVOUS SYSTEM DISORDERS',
    'DIABETES',
    'DISORDERS OF PANCREAS EXCEPT MALIGNANCY',
    'DYSEQUILIBRIUM',
    'ESOPHAGITIS, GASTROENT & MISC DIGEST DISORDERS',
    'EXTRACRANIAL PROCEDURES',
    'FRACTURES OF HIP & PELVIS',
    'FX, SPRN, STRN & DISL EXCEPT FEMUR, HIP, PELVIS & THIGH',
    'G.I. HEMORRHAGE',
    'G.I. OBSTRUCTION',
    'HEART FAILURE & SHOCK',
    'HIP & FEMUR PROCEDURES EXCEPT MAJOR JOINT',
    'HYPERTENSION',
    'INFECTIOUS & PARASITIC DISEASES',
    'INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION',
    'KIDNEY & URINARY TRACT INFECTIONS',
    'LAPAROSCOPIC CHOLECYSTECTOMY',
    'MAJOR CARDIOVASC PROCEDURES',
    'MAJOR GASTROINTESTINAL DISORDERS & PERITONEAL INFECTIONS',
    'MAJOR JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY',
    'MAJOR SMALL & LARGE BOWEL PROCEDURES',
    'MEDICAL BACK PROBLEMS',
    'MISC DISORDERS OF NUTRITION,METABOLISM,FLUIDS/ELECTROLYTES',
    'OTHER CIRCULATORY SYSTEM DIAGNOSES',
    'OTHER DIGESTIVE SYSTEM DIAGNOSES',
    'OTHER KIDNEY & URINARY TRACT DIAGNOSES',
    'OTHER VASCULAR PROCEDURES',
    'PERC CARDIOVASC PROC',
    'PERIPHERAL VASCULAR DISORDERS',
    'PERMANENT CARDIAC PACEMAKER IMPLANT',
    'POISONING & TOXIC EFFECTS OF DRUGS',
    'PSYCHOSES',
    'PULMONARY EDEMA & RESPIRATORY FAILURE',
    'PULMONARY EMBOLISM',
    'RED BLOOD CELL DISORDERS',
    'RENAL FAILURE',
    'RESPIRATORY INFECTIONS & INFLAMMATIONS',
    'RESPIRATORY SYSTEM DIAGNOSIS',
    'SEIZURES',
    'SEPTICEMIA OR SEVERE SEPSIS',
    'SIGNS & SYMPTOMS',
    'SIMPLE PNEUMONIA & PLEURISY',
    'SPINAL FUSION EXCEPT CERVICAL',
    'SYNCOPE & COLLAPSE',
    "ratio_to_max_payment"]
    return df[select]
 
def race_data(df):
     
    select = ['SEX_AND_AGE_Total_population',"Percent_RACE_Total_population_One_race",
    "Percent_RACE_Total_population_Two_or_more_races",
    "Percent_RACE_Total_population_One_race.1",
    "Percent_RACE_Total_population_One_race_White",
    "Percent_RACE_Total_population_One_race_Black_or_African_American",
    "Percent_RACE_Total_population_One_race_American_Indian_and_Alaska_Native",
    "Percent_RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Cherokee_tribal_grouping",
    "Percent_RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Chippewa_tribal_grouping",
    "Percent_RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Navajo_tribal_grouping",
    "Percent_RACE_Total_population_One_race_American_Indian_and_Alaska_Native_Sioux_tribal_grouping",
    "Percent_RACE_Total_population_One_race_Asian",
    "Percent_RACE_Total_population_One_race_Asian_Asian_Indian",
    "Percent_RACE_Total_population_One_race_Asian_Chinese",
    "Percent_RACE_Total_population_One_race_Asian_Filipino",
    "Percent_RACE_Total_population_One_race_Asian_Japanese",
    "Percent_RACE_Total_population_One_race_Asian_Korean",
    "Percent_RACE_Total_population_One_race_Asian_Vietnamese",
    "Percent_RACE_Total_population_One_race_Asian_Other_Asian",
    "Percent_RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander",
    "Percent_RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Native_Hawaiian",
    "Percent_RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Guamanian_or_Chamorro",
    "Percent_RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Samoan",
    "Percent_RACE_Total_population_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Other_Pacific_Islander",
    "Percent_RACE_Total_population_One_race_Some_other_race",
    "Percent_RACE_Total_population_Two_or_more_races.1",
    "Percent_RACE_Total_population_Two_or_more_races_White_and_Black_or_African_American",
    "Percent_RACE_Total_population_Two_or_more_races_White_and_American_Indian_and_Alaska_Native",
    "Percent_RACE_Total_population_Two_or_more_races_White_and_Asian",
    "Percent_RACE_Total_population_Two_or_more_races_Black_or_African_American_and_American_Indian_and_Alaska_Native",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_White",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Black_or_African_American",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_American_Indian_and_Alaska_Native",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Asian",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Native_Hawaiian_and_Other_Pacific_Islander",
    "Percent_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population_Some_other_race",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Mexican",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Puerto_Rican",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Cuban",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Hispanic_or_Latino_(of_any_race)_Other_Hispanic_or_Latino",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_White_alone",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Black_or_African_American_alone",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_American_Indian_and_Alaska_Native_alone",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Asian_alone",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Native_Hawaiian_and_Other_Pacific_Islander_alone",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Some_other_race_alone",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Two_or_more_races",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Two_or_more_races_Two_races_including_Some_other_race",
    "Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population_Not_Hispanic_or_Latino_Two_or_more_races_Two_races_excluding_Some_other_race_and_Three_or_more_races",
    "Total_housing_units",
    'ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE',
    'ALCOHOL/DRUG ABUSE OR DEPENDENCE',
    'ATHEROSCLEROSIS',
    'BACK & NECK PROC EXC SPINAL FUSION',
    'BRONCHITIS & ASTHMA',
    'CARDIAC ARRHYTHMIA & CONDUCTION DISORDERS',
    'CELLULITIS',
    'CERVICAL SPINAL FUSION',
    'CHEST PAIN',
    'CHRONIC OBSTRUCTIVE PULMONARY DISEASE',
    'CIRCULATORY DISORDERS EXCEPT AMI,',
    'CRANIAL & PERIPHERAL NERVE DISORDERS',
    'DEGENERATIVE NERVOUS SYSTEM DISORDERS',
    'DIABETES',
    'DISORDERS OF PANCREAS EXCEPT MALIGNANCY',
    'DYSEQUILIBRIUM',
    'ESOPHAGITIS, GASTROENT & MISC DIGEST DISORDERS',
    'EXTRACRANIAL PROCEDURES',
    'FRACTURES OF HIP & PELVIS',
    'FX, SPRN, STRN & DISL EXCEPT FEMUR, HIP, PELVIS & THIGH',
    'G.I. HEMORRHAGE',
    'G.I. OBSTRUCTION',
    'HEART FAILURE & SHOCK',
    'HIP & FEMUR PROCEDURES EXCEPT MAJOR JOINT',
    'HYPERTENSION',
    'INFECTIOUS & PARASITIC DISEASES',
    'INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION',
    'KIDNEY & URINARY TRACT INFECTIONS',
    'LAPAROSCOPIC CHOLECYSTECTOMY',
    'MAJOR CARDIOVASC PROCEDURES',
    'MAJOR GASTROINTESTINAL DISORDERS & PERITONEAL INFECTIONS',
    'MAJOR JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY',
    'MAJOR SMALL & LARGE BOWEL PROCEDURES',
    'MEDICAL BACK PROBLEMS',
    'MISC DISORDERS OF NUTRITION,METABOLISM,FLUIDS/ELECTROLYTES',
    'OTHER CIRCULATORY SYSTEM DIAGNOSES',
    'OTHER DIGESTIVE SYSTEM DIAGNOSES',
    'OTHER KIDNEY & URINARY TRACT DIAGNOSES',
    'OTHER VASCULAR PROCEDURES',
    'PERC CARDIOVASC PROC',
    'PERIPHERAL VASCULAR DISORDERS',
    'PERMANENT CARDIAC PACEMAKER IMPLANT',
    'POISONING & TOXIC EFFECTS OF DRUGS',
    'PSYCHOSES',
    'PULMONARY EDEMA & RESPIRATORY FAILURE',
    'PULMONARY EMBOLISM',
    'RED BLOOD CELL DISORDERS',
    'RENAL FAILURE',
    'RESPIRATORY INFECTIONS & INFLAMMATIONS',
    'RESPIRATORY SYSTEM DIAGNOSIS',
    'SEIZURES',
    'SEPTICEMIA OR SEVERE SEPSIS',
    'SIGNS & SYMPTOMS',
    'SIMPLE PNEUMONIA & PLEURISY',
    'SPINAL FUSION EXCEPT CERVICAL',
    'SYNCOPE & COLLAPSE',
    "ratio_to_max_payment"]
    return df[select]

def poverty_data(df):
     
    select = ['SEX_AND_AGE_Total_population',"Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years_18_to_34_years",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_18_to_64_years_35_to_64_years",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_60_years_and_over",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_AGE_65_years_and_over",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_SEX_Male",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_SEX_Female",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_RACE_AND_HISPANIC_OR_LATINO_ORIGIN_White_alone",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Less_than_high_school_graduate",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_High_school_graduate_(includes_equivalency)",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Some_college,_associate's_degree",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EDUCATIONAL_ATTAINMENT_Population_25_years_and_over_Bachelor's_degree_or_higher",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed_Male",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_EMPLOYMENT_STATUS_Civilian_labor_force_16_years_and_over_Employed_Female",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Worked_full-time,_year-round_in_the_past_12_months",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Worked_part-time_or_part-year_in_the_past_12_months",
    "Percent_below_poverty_level_Population_for_whom_poverty_status_is_determined_WORK_EXPERIENCE_Population_16_years_and_over_Did_not_work",
    "Percent_below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED",
    "Percent_below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Male",
    "Percent_below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Female",
    "Percent_below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Worked_full-time,_year-round_in_the_past_12_months",
    "Percent_below_poverty_level_UNRELATED_INDIVIDUALS_FOR_WHOM_POVERTY_STATUS_IS_DETERMINED_Did_not_work",
    'ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE',
    'ALCOHOL/DRUG ABUSE OR DEPENDENCE',
    'ATHEROSCLEROSIS',
    'BACK & NECK PROC EXC SPINAL FUSION',
    'BRONCHITIS & ASTHMA',
    'CARDIAC ARRHYTHMIA & CONDUCTION DISORDERS',
    'CELLULITIS',
    'CERVICAL SPINAL FUSION',
    'CHEST PAIN',
    'CHRONIC OBSTRUCTIVE PULMONARY DISEASE',
    'CIRCULATORY DISORDERS EXCEPT AMI,',
    'CRANIAL & PERIPHERAL NERVE DISORDERS',
    'DEGENERATIVE NERVOUS SYSTEM DISORDERS',
    'DIABETES',
    'DISORDERS OF PANCREAS EXCEPT MALIGNANCY',
    'DYSEQUILIBRIUM',
    'ESOPHAGITIS, GASTROENT & MISC DIGEST DISORDERS',
    'EXTRACRANIAL PROCEDURES',
    'FRACTURES OF HIP & PELVIS',
    'FX, SPRN, STRN & DISL EXCEPT FEMUR, HIP, PELVIS & THIGH',
    'G.I. HEMORRHAGE',
    'G.I. OBSTRUCTION',
    'HEART FAILURE & SHOCK',
    'HIP & FEMUR PROCEDURES EXCEPT MAJOR JOINT',
    'HYPERTENSION',
    'INFECTIOUS & PARASITIC DISEASES',
    'INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION',
    'KIDNEY & URINARY TRACT INFECTIONS',
    'LAPAROSCOPIC CHOLECYSTECTOMY',
    'MAJOR CARDIOVASC PROCEDURES',
    'MAJOR GASTROINTESTINAL DISORDERS & PERITONEAL INFECTIONS',
    'MAJOR JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY',
    'MAJOR SMALL & LARGE BOWEL PROCEDURES',
    'MEDICAL BACK PROBLEMS',
    'MISC DISORDERS OF NUTRITION,METABOLISM,FLUIDS/ELECTROLYTES',
    'OTHER CIRCULATORY SYSTEM DIAGNOSES',
    'OTHER DIGESTIVE SYSTEM DIAGNOSES',
    'OTHER KIDNEY & URINARY TRACT DIAGNOSES',
    'OTHER VASCULAR PROCEDURES',
    'PERC CARDIOVASC PROC',
    'PERIPHERAL VASCULAR DISORDERS',
    'PERMANENT CARDIAC PACEMAKER IMPLANT',
    'POISONING & TOXIC EFFECTS OF DRUGS',
    'PSYCHOSES',
    'PULMONARY EDEMA & RESPIRATORY FAILURE',
    'PULMONARY EMBOLISM',
    'RED BLOOD CELL DISORDERS',
    'RENAL FAILURE',
    'RESPIRATORY INFECTIONS & INFLAMMATIONS',
    'RESPIRATORY SYSTEM DIAGNOSIS',
    'SEIZURES',
    'SEPTICEMIA OR SEVERE SEPSIS',
    'SIGNS & SYMPTOMS',
    'SIMPLE PNEUMONIA & PLEURISY',
    'SPINAL FUSION EXCEPT CERVICAL',
    'SYNCOPE & COLLAPSE',
    "ratio_to_max_payment"]
    return df[select]

def income_data(df):
     
    select = ['SEX_AND_AGE_Total_population',"Households_Total",
    "Households_Less_than_$10,000",
    "Households_$10,000_to_$14,999",
    "Households_$15,000_to_$24,999",
    "Households_$25,000_to_$34,999",
    "Households_$35,000_to_$49,999",
    "Households_$50,000_to_$74,999",
    "Households_$100,000_to_$149,999",
    "Households_$150,000_to_$199,999",
    "Households_$200,000_or_more",
    "Households_Median_income_(dollars)",
    "Households_Mean_income_(dollars)",
    "Households_PERCENT_IMPUTED_Household_income_in_the_past_12_months",
    "Families_Total",
    "Families_Less_than_$10,000",
    "Families_$10,000_to_$14,999",
    "Families_$15,000_to_$24,999",
    "Families_$25,000_to_$34,999",
    "Families_$35,000_to_$49,999",
    "Families_$50,000_to_$74,999",
    "Families_$75,000_to_$99,999",
    "Families_$100,000_to_$149,999",
    "Families_$150,000_to_$199,999",
    "Families_$200,000_or_more",
    "Families_Median_income_(dollars)",
    "Families_Mean_income_(dollars)",
    "Families_PERCENT_IMPUTED_Family_income_in_the_past_12_months",
    "Married-couple_families_Median_income_(dollars)",
    "Households_$75,000_to_$99,999",
    "Nonfamily_households_Median_income_(dollars)",
    "Nonfamily_households_PERCENT_IMPUTED_Nonfamily_income_in_the_past_12_months",
    'ACUTE MYOCARDIAL INFARCTION, DISCHARGED ALIVE',
    'ALCOHOL/DRUG ABUSE OR DEPENDENCE',
    'ATHEROSCLEROSIS',
    'BACK & NECK PROC EXC SPINAL FUSION',
    'BRONCHITIS & ASTHMA',
    'CARDIAC ARRHYTHMIA & CONDUCTION DISORDERS',
    'CELLULITIS',
    'CERVICAL SPINAL FUSION',
    'CHEST PAIN',
    'CHRONIC OBSTRUCTIVE PULMONARY DISEASE',
    'CIRCULATORY DISORDERS EXCEPT AMI,',
    'CRANIAL & PERIPHERAL NERVE DISORDERS',
    'DEGENERATIVE NERVOUS SYSTEM DISORDERS',
    'DIABETES',
    'DISORDERS OF PANCREAS EXCEPT MALIGNANCY',
    'DYSEQUILIBRIUM',
    'ESOPHAGITIS, GASTROENT & MISC DIGEST DISORDERS',
    'EXTRACRANIAL PROCEDURES',
    'FRACTURES OF HIP & PELVIS',
    'FX, SPRN, STRN & DISL EXCEPT FEMUR, HIP, PELVIS & THIGH',
    'G.I. HEMORRHAGE',
    'G.I. OBSTRUCTION',
    'HEART FAILURE & SHOCK',
    'HIP & FEMUR PROCEDURES EXCEPT MAJOR JOINT',
    'HYPERTENSION',
    'INFECTIOUS & PARASITIC DISEASES',
    'INTRACRANIAL HEMORRHAGE OR CEREBRAL INFARCTION',
    'KIDNEY & URINARY TRACT INFECTIONS',
    'LAPAROSCOPIC CHOLECYSTECTOMY',
    'MAJOR CARDIOVASC PROCEDURES',
    'MAJOR GASTROINTESTINAL DISORDERS & PERITONEAL INFECTIONS',
    'MAJOR JOINT REPLACEMENT OR REATTACHMENT OF LOWER EXTREMITY',
    'MAJOR SMALL & LARGE BOWEL PROCEDURES',
    'MEDICAL BACK PROBLEMS',
    'MISC DISORDERS OF NUTRITION,METABOLISM,FLUIDS/ELECTROLYTES',
    'OTHER CIRCULATORY SYSTEM DIAGNOSES',
    'OTHER DIGESTIVE SYSTEM DIAGNOSES',
    'OTHER KIDNEY & URINARY TRACT DIAGNOSES',
    'OTHER VASCULAR PROCEDURES',
    'PERC CARDIOVASC PROC',
    'PERIPHERAL VASCULAR DISORDERS',
    'PERMANENT CARDIAC PACEMAKER IMPLANT',
    'POISONING & TOXIC EFFECTS OF DRUGS',
    'PSYCHOSES',
    'PULMONARY EDEMA & RESPIRATORY FAILURE',
    'PULMONARY EMBOLISM',
    'RED BLOOD CELL DISORDERS',
    'RENAL FAILURE',
    'RESPIRATORY INFECTIONS & INFLAMMATIONS',
    'RESPIRATORY SYSTEM DIAGNOSIS',
    'SEIZURES',
    'SEPTICEMIA OR SEVERE SEPSIS',
    'SIGNS & SYMPTOMS',
    'SIMPLE PNEUMONIA & PLEURISY',
    'SPINAL FUSION EXCEPT CERVICAL',
    'SYNCOPE & COLLAPSE',
    "ratio_to_max_payment"]
    return df[select]

def random_forest(data, multiplier=2, target_column='ratio_to_max_payment', show=20):
    y= data.ratio_to_max_payment >= multiplier*data.ratio_to_max_payment.std()+data.ratio_to_max_payment.mean()
    X = data.drop(columns=[target_column])
    data_train, data_test, target_train, target_test = train_test_split(X, y, 
                                                                        test_size = 0.25)


    forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5)
    forest.fit(data_train,target_train)
    df_feature_importance = pd.Series(data = forest.feature_importances_, index= X.columns).sort_values(ascending=False)
    print(df_feature_importance[:show])
    print(forest.score(data_train,target_train), forest.score(data_test,target_test))
    return forest, df_feature_importance

