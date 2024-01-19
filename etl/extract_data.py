import os
import datetime
import pickle
import glob

import pandas as pd

from accdbtools import accdb2csv
from get_schema import get_schema_db, get_schema_orgnized



# USER INPUT
SELECTED_STATUS = 'Actif Min.' # all, 
PREPROCESS_MODE = 'all' # one: preprocess selected category, all: preprocess everything
SELECTED_ELEMENT_CAT = "Poutre"
INTERVENTIONS_FLAG = 2 # everything: 1, only deterioration (without interventions): 2, only interventions: 3

# get schema 
(schema_inspections, schema_structures,
schema_interventions, schema_inspectors) = get_schema_db()

# read data
current_path = os.getcwd()
csv_path = os.path.join(current_path, "*.csv")
csv_files = glob.glob(csv_path)

if len(csv_files) < 8 :
    db_path = os.path.join(current_path, "db_files/*.accdb")
    accdb_files = glob.glob(db_path)
    latest_db = max(accdb_files, key=os.path.getctime)
    db_path = os.path.join(current_path, latest_db)

    read_accdb = accdb2csv(db_path)

# prepare path [file_names = names of tables inside accdb file]
inspections_path = os.path.join(current_path, "*elements_inspec*.csv")
inspectors_path = os.path.join(current_path, "*ingenieur_responsable*.csv")
structures_path = os.path.join(current_path, "*inventaire_inspect*.csv")
details_path = os.path.join(current_path, "*inventaire_detail*.csv")
interventions_path = os.path.join(current_path, "*travaux_realis*.csv")

# read files
inspections_db = pd.read_csv(glob.glob(inspections_path)[0])
inspectors_db = pd.read_csv(glob.glob(inspectors_path)[0])
structures_db = pd.read_csv(glob.glob(structures_path)[0])
detials_db = pd.read_csv(glob.glob(details_path)[0])
interventions_db = pd.read_csv(glob.glob(interventions_path)[0])

#IB=AppLink.TrainingApp.IntBudgetDataStore;

if PREPROCESS_MODE == 'all':
    cat_list = inspections_db.ELEMENT.unique()
else:
    cat_list = [SELECTED_ELEMENT_CAT]

inspections_db = inspections_db[schema_inspections]
interventions_db = interventions_db[schema_interventions]
structures_db = structures_db[schema_structures]
inspectors_db = inspectors_db[schema_inspectors]

# preprocess structures db
#structures_db['NewID'] = structures_db[schema_structures[0]].factorize()[0]
#mapping_id = structures_db.set_index(schema_structures[0])['NewID'].to_dict()

structures_df = structures_db.__deepcopy__()
#structures_df[schema_structures[0]] = structures_db['NewID']
#structures_df.drop(columns=['NewID'], inplace=True)

structures_df['AGE'] = datetime.date.today().year - structures_df[schema_structures[9]]

if SELECTED_STATUS != 'all':
    structures_df = structures_df[structures_df[schema_structures[10]] == SELECTED_STATUS]

for element in cat_list:
    # get the orgnized data schema
    schema_orgnized_data = get_schema_orgnized(element, schema_inspections)

    # filter by element category 
    inspections_elem_df = inspections_db[inspections_db.ELEMENT == element]
    # Map the values from structure id to new structure id
    #inspections_elem_df[schema_inspections[0]] = inspections_elem_df[schema_inspections[0]].map(mapping_id)

    interventions_elem_df = interventions_db[interventions_db.ELEMENT == element]
    # Map the values from structure id to new structure id
    #interventions_elem_df[schema_interventions[0]] = interventions_elem_df[schema_interventions[0]].map(mapping_id)

    # check interventions
    unique_interventions = interventions_elem_df.drop_duplicates().reset_index(drop=True)

    # find matching indicies 
    # get common indices
    elem_intersection_indices = inspections_elem_df[inspections_elem_df[schema_inspections[14]].isin(unique_interventions[schema_interventions[4]])].index
    if elem_intersection_indices.empty:
        elem_intersection_indices = inspections_elem_df[inspections_elem_df[schema_inspections[0]].isin(unique_interventions[schema_interventions[0]])].index
        if ~elem_intersection_indices.empty:
            no_element_id = elem_intersection_indices

    interventions_intersection_indices = unique_interventions[unique_interventions[schema_interventions[4]].isin(inspections_elem_df[schema_inspections[14]])].index

    # Find indices where 'IDE_ELEMN' in inspections is NOT in 'IDENTIFIANT ELEMENT' in intervetnions
    uncommon_indices_1 = inspections_elem_df[~inspections_elem_df[schema_inspections[14]].isin(unique_interventions[schema_interventions[4]])].index

    # Find indices where 'IDENTIFIANT ELEMENT' in intervetnions is NOT in 'IDE_ELEMN' in inspections
    uncommon_indices_2 = unique_interventions[~unique_interventions[schema_interventions[4]].isin(inspections_elem_df[schema_inspections[14]])].index

    # deterioration only
    if INTERVENTIONS_FLAG == 2 :
        inspections_elem_df = inspections_elem_df.loc[uncommon_indices_1]

    # interventions only
    elif INTERVENTIONS_FLAG == 3 :
        inspections_elem_df = inspections_elem_df.loc[elem_intersection_indices]
        unique_interventions = unique_interventions.loc[interventions_intersection_indices]

    if ~inspections_elem_df.empty:
        material_categories = inspections_elem_df[schema_inspections[6]].unique()
        type_element_categories = inspections_elem_df[schema_inspections[7]].unique()

        inspections_elem_df[schema_inspections[6]] = inspections_elem_df[schema_inspections[6]].factorize()[0]
        inspections_elem_df[schema_inspections[7]] = inspections_elem_df[schema_inspections[7]].factorize()[0]

        # preprocess inspectors
        mapping_dict = inspectors_db.set_index(schema_inspectors[0])[schema_inspectors[1]].to_dict()
        # Map the values from inspections id to inspectors id
        inspections_elem_df[schema_inspections[13]] = inspections_elem_df[schema_inspections[13]].map(mapping_dict)
        # rename the column
        inspections_elem_df = inspections_elem_df.rename(columns={schema_inspections[13]: schema_inspectors[1]})

        # compute the aggregate metric y
        inspections_elem_df['Y'] = (inspections_elem_df[schema_inspections[8]] + 0.75 * inspections_elem_df[schema_inspections[9]]
                                + 0.5 * inspections_elem_df[schema_inspections[10]] + 0.25 * inspections_elem_df[schema_inspections[11]])
        
        # merge structures and inspections dataframes
        inspections_with_attributes = pd.merge(inspections_elem_df, structures_df, on=schema_inspections[0], how='left')

        # change datetime to year
        inspections_with_attributes[schema_inspections[1]] = pd.to_datetime(inspections_with_attributes[schema_inspections[1]]).dt.year

        if INTERVENTIONS_FLAG == 3 :
            unique_interventions = unique_interventions.drop(unique_interventions.columns[[0, 3]], axis=1)
            # pre-process activity code
            unique_interventions[schema_interventions[2]] = unique_interventions[schema_interventions[2]].astype(str).str[0].astype(int)
            # merge with inspections
            inspections_with_attributes = pd.merge(inspections_with_attributes, unique_interventions, left_on=schema_inspections[14], right_on=schema_interventions[4], how='left')
            inspections_with_attributes = inspections_with_attributes.drop(schema_interventions[4], axis=1)
            # update schema
            schema_orgnized_data.extend([schema_interventions[1], schema_interventions[2]])

        # get inspectors
        inspectors_id = inspections_with_attributes[schema_inspectors[1]].dropna().unique().astype(int)

        grouped_df = inspections_with_attributes.groupby([schema_structures[0], schema_inspections[-1]])
        # Pivot the DataFrame using pivot_table, collecting values into lists
        # pivoted_df = grouped_df.pivot_table(index=schema_structures[0], columns=schema_inspections[-1], values=schema_orgnized_data, aggfunc=lambda x: list(x))
        gb = grouped_df.groups
        training_data = {}
        for key, values in gb.items():
            df_store = inspections_with_attributes.iloc[values]
            df_store = df_store[schema_orgnized_data].reset_index(drop=True)
            df_store = df_store.sort_values(schema_orgnized_data[6])
            training_data[key] = df_store

        # save files
        if INTERVENTIONS_FLAG == 3 :
            with open('pkl_files/%s_interventions_trainingdata.pkl'%(element.replace('/','-')), 'wb') as handle:
                pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('pkl_files/%s_interventions_inspectors.pkl'%(element.replace('/','-')), 'wb') as handle:
                pickle.dump(inspectors_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('pkl_files/%s_interventions_metadata.pkl'%(element.replace('/','-')), 'wb') as handle:
                pickle.dump([material_categories, type_element_categories, schema_orgnized_data], 
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            with open('pkl_files/%s_trainingdata.pkl'%(element.replace('/','-')), 'wb') as handle:
                pickle.dump(training_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('pkl_files/%s_inspectors.pkl'%(element.replace('/','-')), 'wb') as handle:
                pickle.dump(inspectors_id, handle, protocol=pickle.HIGHEST_PROTOCOL)

            with open('pkl_files/%s_metadata.pkl'%(element.replace('/','-')), 'wb') as handle:
                pickle.dump([material_categories, type_element_categories, schema_orgnized_data], 
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
