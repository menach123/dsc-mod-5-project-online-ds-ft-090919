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



class DATA_CLASS(object):
    
    def __init__(self):
        self.poverty_df = upload_county_acs_data('ACSDP5Y2012.DP03_data_with_overlays_2019-12-31T163946.csv')
        self.population_df = upload_county_acs_data('ACSDP5Y2012.DP05_data_with_overlays_2019-12-31T193014.csv')
        df = merge_acs_data( self.poverty_df,self.population_df)
        df = remove_duplicate_countystate(df)
        self.acs_data = df
        self.cms_df = cms_data()
        self.df = self.cms_df.merge(df.drop(columns =['County','State']), how='left', on=['CountyState'])
        self.initial_df = self.cms_df.merge(df.drop(columns =['County','State']), how='left', on=['CountyState'])
        
        self.target = 'ratio_to_max_payment'
        self.target_df = self.df[self.target]
        self.cms_label_df = cms_procedure_dummy_labels(self.cms_df)
        pass
        
    def set_target(self, column):
        """
        Entering the target column
        """
        self.target = column
        self.target_df = self.df[self.target]
        pass

    def numerical_columns(self):
        """
        Separating the numerical columns from the object columns
        """
        self.df, self.object_df = separate_num_columns(self.df)
        pass

    def tree_dataframe(self):
        """
        Dataframe for tree modeling, Target column is separate.
        """
        columns = ['Percent_EMPLOYMENT_STATUS_Population_16_years_and_over',
                    'Percent_EMPLOYMENT_STATUS_Civilian_labor_force',
                    'Percent_EMPLOYMENT_STATUS_Females_16_years_and_over',
                    'Percent_EMPLOYMENT_STATUS_Own_children_under_6_years',
                    'Percent_EMPLOYMENT_STATUS_Own_children_6_to_17_years',
                    'Percent_COMMUTING_TO_WORK_Workers_16_years_and_over',
                    'Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population',
                    'Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population_under_18_years',
                    'Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population_18_to_64_years',
                    'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force',
                    'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Employed',
                    'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Unemployed',
                    'Percent_HEALTH_INSURANCE_COVERAGE_Not_in_labor_force',
                    'Percent_SEX_AND_AGE_18_years_and_over.1',
                    'Percent_SEX_AND_AGE_65_years_and_over.1',
                    'Percent_RACE_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population',
                    'Percent_HISPANIC_OR_LATINO_AND_RACE_Total_population']

        for i in columns:
            self.df[i] = self.df[i]/ self.df['SEX_AND_AGE_Total_population']

        columns = ['max_payment',
        'Percent_EMPLOYMENT_STATUS_Population_16_years_and_over',
        'Percent_EMPLOYMENT_STATUS_In_labor_force',
        'Percent_EMPLOYMENT_STATUS_In_labor_force_Civilian_labor_force',
        'Percent_EMPLOYMENT_STATUS_In_labor_force_Civilian_labor_force_Employed',
        'Percent_EMPLOYMENT_STATUS_In_labor_force_Civilian_labor_force_Unemployed',
        'Percent_EMPLOYMENT_STATUS_In_labor_force_Armed_Forces',
        'Percent_EMPLOYMENT_STATUS_Not_in_labor_force',
        'Percent_EMPLOYMENT_STATUS_Civilian_labor_force',
        'Percent_EMPLOYMENT_STATUS_Percent_Unemployed',
        'Percent_EMPLOYMENT_STATUS_Females_16_years_and_over',
        'Percent_EMPLOYMENT_STATUS_In_labor_force.1',
        'Percent_EMPLOYMENT_STATUS_In_labor_force_Civilian_labor_force.1',
        'Percent_EMPLOYMENT_STATUS_In_labor_force_Civilian_labor_force_Employed.1',
        'Percent_EMPLOYMENT_STATUS_Own_children_under_6_years',
        'Percent_EMPLOYMENT_STATUS_All_parents_in_family_in_labor_force',
        'Percent_EMPLOYMENT_STATUS_Own_children_6_to_17_years',
        'Percent_EMPLOYMENT_STATUS_All_parents_in_family_in_labor_force.1',
        'Percent_COMMUTING_TO_WORK_Workers_16_years_and_over',
        'Percent_COMMUTING_TO_WORK_Car,_truck,_or_van_--_drove_alone',
        'Percent_COMMUTING_TO_WORK_Car,_truck,_or_van_--_carpooled',
        'Percent_COMMUTING_TO_WORK_Public_transportation_(excluding_taxicab)',
        'Percent_COMMUTING_TO_WORK_Walked',
        'Percent_COMMUTING_TO_WORK_Other_means',
        'Percent_COMMUTING_TO_WORK_Worked_at_home',
        'Percent_OCCUPATION_Civilian_employed_population_16_years_and_over',
        'Percent_OCCUPATION_Management,_business,_science,_and_arts_occupations',
        'Percent_OCCUPATION_Service_occupations',
        'Percent_OCCUPATION_Sales_and_office_occupations',
        'Percent_OCCUPATION_Natural_resources,_construction,_and_maintenance_occupations',
        'Percent_OCCUPATION_Production,_transportation,_and_material_moving_occupations',
        'Percent_INDUSTRY_Civilian_employed_population_16_years_and_over',
        'Percent_INDUSTRY_Agriculture,_forestry,_fishing_and_hunting,_and_mining',
        'Percent_INDUSTRY_Construction',
        'Percent_INDUSTRY_Manufacturing',
        'Percent_INDUSTRY_Wholesale_trade',
        'Percent_INDUSTRY_Retail_trade',
        'Percent_INDUSTRY_Transportation_and_warehousing,_and_utilities',
        'Percent_INDUSTRY_Information',
        'Percent_INDUSTRY_Finance_and_insurance,_and_real_estate_and_rental_and_leasing',
        'Percent_INDUSTRY_Professional,_scientific,_and_management,_and_administrative_and_waste_management_services',
        'Percent_INDUSTRY_Educational_services,_and_health_care_and_social_assistance',
        'Percent_INDUSTRY_Arts,_entertainment,_and_recreation,_and_accommodation_and_food_services',
        'Percent_INDUSTRY_Other_services,_except_public_administration',
        'Percent_INDUSTRY_Public_administration',
        'Percent_CLASS_OF_WORKER_Civilian_employed_population_16_years_and_over',
        'Percent_CLASS_OF_WORKER_Private_wage_and_salary_workers',
        'Percent_CLASS_OF_WORKER_Government_workers',
        'Percent_CLASS_OF_WORKER_Self-employed_in_own_not_incorporated_business_workers',
        'Percent_CLASS_OF_WORKER_Unpaid_family_workers',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Total_households',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Less_than_$10,000',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$10,000_to_$14,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$15,000_to_$24,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$25,000_to_$34,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$35,000_to_$49,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$50,000_to_$74,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$75,000_to_$99,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$100,000_to_$149,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$150,000_to_$199,999',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$200,000_or_more',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_With_earnings',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_With_Social_Security',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_With_retirement_income',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_With_Supplemental_Security_Income',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_With_cash_public_assistance_income',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_With_Food_Stamp/SNAP_benefits_in_the_past_12_months',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Families',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Less_than_$10,000.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$10,000_to_$14,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$15,000_to_$24,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$25,000_to_$34,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$35,000_to_$49,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$50,000_to_$74,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$75,000_to_$99,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$100,000_to_$149,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$150,000_to_$199,999.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_$200,000_or_more.1',
        'Percent_INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Nonfamily_households',
        'Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population',
        'Percent_HEALTH_INSURANCE_COVERAGE_With_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_With_health_insurance_coverage_With_private_health_insurance',
        'Percent_HEALTH_INSURANCE_COVERAGE_With_health_insurance_coverage_With_public_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_No_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population_under_18_years',
        'Percent_HEALTH_INSURANCE_COVERAGE_No_health_insurance_coverage.1',
        'Percent_HEALTH_INSURANCE_COVERAGE_Civilian_noninstitutionalized_population_18_to_64_years',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Employed',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Employed_With_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Employed_With_health_insurance_coverage_With_private_health_insurance',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Employed_With_health_insurance_coverage_With_public_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Employed_No_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Unemployed',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Unemployed_With_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Unemployed_With_health_insurance_coverage_With_private_health_insurance',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Unemployed_With_health_insurance_coverage_With_public_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_In_labor_force_Unemployed_No_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_Not_in_labor_force',
        'Percent_HEALTH_INSURANCE_COVERAGE_Not_in_labor_force_With_health_insurance_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_Not_in_labor_force_With_health_insurance_coverage_With_private_health_insurance',
        'Percent_HEALTH_INSURANCE_COVERAGE_Not_in_labor_force_With_health_insurance_coverage_With_public_coverage',
        'Percent_HEALTH_INSURANCE_COVERAGE_Not_in_labor_force_No_health_insurance_coverage',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_families',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_families_With_related_children_under_18_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_families_With_related_children_under_18_years_With_related_children_under_5_years_only',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Married_couple_families',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Married_couple_families_With_related_children_under_18_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Married_couple_families_With_related_children_under_18_years_With_related_children_under_5_years_only',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Families_with_female_householder,_no_husband_present',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Families_with_female_householder,_no_husband_present_With_related_children_under_18_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Families_with_female_householder,_no_husband_present_With_related_children_under_18_years_With_related_children_under_5_years_only',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_All_people',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Under_18_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Under_18_years_Related_children_under_18_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Under_18_years_Related_children_under_18_years_Related_children_under_5_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Under_18_years_Related_children_under_18_years_Related_children_5_to_17_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_18_years_and_over',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_18_years_and_over_18_to_64_years',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_18_years_and_over_65_years_and_over',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_People_in_families',
        'Percent_PERCENTAGE_OF_FAMILIES_AND_PEOPLE_WHOSE_INCOME_IN_THE_PAST_12_MONTHS_IS_BELOW_THE_POVERTY_LEVEL_Unrelated_individuals_15_years_and_over',
        'Percent_SEX_AND_AGE_Total_population',
        'Percent_SEX_AND_AGE_Male',
        'Percent_SEX_AND_AGE_Female',
        'Percent_SEX_AND_AGE_Under_5_years',
        'Percent_SEX_AND_AGE_5_to_9_years',
        'Percent_SEX_AND_AGE_10_to_14_years',
        'Percent_SEX_AND_AGE_15_to_19_years',
        'Percent_SEX_AND_AGE_20_to_24_years',
        'Percent_SEX_AND_AGE_25_to_34_years',
        'Percent_SEX_AND_AGE_35_to_44_years',
        'Percent_SEX_AND_AGE_45_to_54_years',
        'Percent_SEX_AND_AGE_55_to_59_years',
        'Percent_SEX_AND_AGE_60_to_64_years',
        'Percent_SEX_AND_AGE_65_to_74_years',
        'Percent_SEX_AND_AGE_75_to_84_years',
        'Percent_SEX_AND_AGE_85_years_and_over',
        'Percent_SEX_AND_AGE_18_years_and_over',
        'Percent_SEX_AND_AGE_21_years_and_over',
        'Percent_SEX_AND_AGE_62_years_and_over',
        'Percent_SEX_AND_AGE_65_years_and_over',
        'Percent_SEX_AND_AGE_18_years_and_over.1',
        'Percent_SEX_AND_AGE_Male.1',
        'Percent_SEX_AND_AGE_Female.1',
        'Percent_SEX_AND_AGE_65_years_and_over.1',
        'Percent_SEX_AND_AGE_Male.2',
        'Percent_SEX_AND_AGE_Female.2',
        'Percent_RACE_One_race',
        'Percent_RACE_Two_or_more_races',
        'Percent_RACE_One_race.1',
        'Percent_RACE_One_race_White',
        'Percent_RACE_One_race_Black_or_African_American',
        'Percent_RACE_One_race_American_Indian_and_Alaska_Native',
        'Percent_RACE_One_race_American_Indian_and_Alaska_Native_Cherokee_tribal_grouping',
        'Percent_RACE_One_race_American_Indian_and_Alaska_Native_Chippewa_tribal_grouping',
        'Percent_RACE_One_race_American_Indian_and_Alaska_Native_Navajo_tribal_grouping',
        'Percent_RACE_One_race_American_Indian_and_Alaska_Native_Sioux_tribal_grouping',
        'Percent_RACE_One_race_Asian',
        'Percent_RACE_One_race_Asian_Asian_Indian',
        'Percent_RACE_One_race_Asian_Chinese',
        'Percent_RACE_One_race_Asian_Filipino',
        'Percent_RACE_One_race_Asian_Japanese',
        'Percent_RACE_One_race_Asian_Korean',
        'Percent_RACE_One_race_Asian_Vietnamese',
        'Percent_RACE_One_race_Asian_Other_Asian',
        'Percent_RACE_One_race_Native_Hawaiian_and_Other_Pacific_Islander',
        'Percent_RACE_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Native_Hawaiian',
        'Percent_RACE_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Guamanian_or_Chamorro',
        'Percent_RACE_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Samoan',
        'Percent_RACE_One_race_Native_Hawaiian_and_Other_Pacific_Islander_Other_Pacific_Islander',
        'Percent_RACE_One_race_Some_other_race',
        'Percent_RACE_Two_or_more_races.1',
        'Percent_RACE_Two_or_more_races_White_and_Black_or_African_American',
        'Percent_RACE_Two_or_more_races_White_and_American_Indian_and_Alaska_Native',
        'Percent_RACE_Two_or_more_races_White_and_Asian',
        'Percent_RACE_Two_or_more_races_Black_or_African_American_and_American_Indian_and_Alaska_Native',
        'Percent_RACE_Race_alone_or_in_combination_with_one_or_more_other_races_Total_population',
        'Percent_RACE_White',
        'Percent_RACE_Black_or_African_American',
        'Percent_RACE_American_Indian_and_Alaska_Native',
        'Percent_RACE_Asian',
        'Percent_RACE_Native_Hawaiian_and_Other_Pacific_Islander',
        'Percent_RACE_Some_other_race',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Hispanic_or_Latino_(of_any_race)',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Hispanic_or_Latino_(of_any_race)_Mexican',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Hispanic_or_Latino_(of_any_race)_Puerto_Rican',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Hispanic_or_Latino_(of_any_race)_Cuban',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Hispanic_or_Latino_(of_any_race)_Other_Hispanic_or_Latino',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_White_alone',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Black_or_African_American_alone',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_American_Indian_and_Alaska_Native_alone',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Asian_alone',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Native_Hawaiian_and_Other_Pacific_Islander_alone',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Some_other_race_alone',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Two_or_more_races',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Two_or_more_races_Two_races_including_Some_other_race',
        'Percent_HISPANIC_OR_LATINO_AND_RACE_Not_Hispanic_or_Latino_Two_or_more_races_Two_races_excluding_Some_other_race,_and_Three_or_more_races',
        'COMMUTING_TO_WORK_Mean_travel_time_to_work_(minutes)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Median_household_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Mean_household_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Median_family_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Mean_family_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Per_capita_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Median_nonfamily_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Mean_nonfamily_income_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Median_earnings_for_workers_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Median_earnings_for_male_full-time,_year-round_workers_(dollars)',
        'INCOME_AND_BENEFITS_(IN_2012_INFLATION-ADJUSTED_DOLLARS)_Median_earnings_for_female_full-time,_year-round_workers_(dollars)',
        'SEX_AND_AGE_Total_population',
        'SEX_AND_AGE_Median_age_(years)',
        'RACE_One_race.1',
        'Total_housing_units']
        for i in self.cms_label_df.columns:
            columns.append(i)   
        self.tree_df = self.df[columns]
        
        pass

    def provider_grouped_df(self, min_procedure=0):
        """
        Dataframes for clustering grouping by provider with the CMS label mulitpled by the target
        Output- labels mulipied by target, adding CMS data
        """
        label_df = self.cms_label_df
        for i in label_df.columns:
            label_df[i] = label_df[i]* self.cms_df.ratio_to_max_payment
        label_df['Provider_Id'] = self.cms_df['Provider_Id']
                
        self.provider_index = self.cms_df.Provider_Id.value_counts().loc[self.cms_df.Provider_Id.value_counts()>= min_procedure].index
        self.provider_label = label_df.groupby('Provider_Id').max()
        self.provider_full_label = self.provider_label.merge(self.df.drop(columns =['County','State']), how='left', on=['CountyState'])
        pass
