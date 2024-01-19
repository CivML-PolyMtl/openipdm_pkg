# Any change in the schema (e.g., order of columns) must be accomodated in the main file {extract_data.py}
def get_schema_orgnized(element, schema_inspections):
    
    list_type = ['Acier structural - pout. triang./arc',
                'Acier structural - tablier','Acier structural - unités de fondation',
                'Appareils d''appui','Assemblages','Autres éléments','Platelage']
    
    list_no_LongTablier = ['Chasse-roue / trottoir','Dessous de la dalle/voûte',
                           'Mur','Mur de tête','Mur en aile','Murs/naiss.voûte/coins infér.',
                        'Plafond suspendu - Tuiles','Revêtement de mur','Voûte / Dalle']
    
    list_no_LargHorstout = ['Dessous de la dalle/voûte','Mur','Mur de tête','Mur en aile',
                            'Murs/naiss.voûte/coins infér.']
    
    list_no_LargCarrossable = ['Mur']

    list_no_DJMA = ['Mur']

    list_no_camions = ['Mur']

    list_no_NBREVOIES = ['Tirants','Toiture']

    # Element - specific options
    # index 7: material column (or no categorical data), index 8: element type column
    if element in list_type:
        CAT_DATA_IND = 8
    else:
        CAT_DATA_IND = 7 

    schema_orgnized_data = ['%A', '%B', '%C', '%D', 'Y', 'IDENTIFIANT INSPECTEUR', 
                            'DATE INSP', 'NO STRUCT', schema_inspections[CAT_DATA_IND], 
                            'AGE','NO TRAV', 'NO ELEM', 'POSITION','LAT','LONG', 
                            'LONG TOTALE']

    if element not in list_no_LongTablier:
        schema_orgnized_data.append('LONG TABLIER')
    
    if element not in list_no_LargHorstout:
        schema_orgnized_data.append('LARG HORS TOUT')

    if element not in list_no_LargCarrossable:
        schema_orgnized_data.append('LARG CARROSSABLE')
    
    schema_orgnized_data.append('SUPERF TABLIER')

    if element not in list_no_DJMA:
        schema_orgnized_data.append('DJMA')

    if element not in list_no_camions:
        schema_orgnized_data.append('% CAMIONS')

    if element not in list_no_NBREVOIES:
        schema_orgnized_data.append('NBRE VOIES')
    
    return schema_orgnized_data

def get_schema_db():
    # schema for each table
    schema_inspections = ['NO STRUCT','DATE INSP','NO TRAV','ELEMENT','NO ELEM',
                        'POSITION','MATERIAU','TYPE ELEMENT','%A','%B',
                        '%C','%D','CEC','IDENTIFIANT INSPECTION','IDE_ELEMN']

    schema_structures = ['NO STRUCT', 'TYPE STRUCTURE', 'LAT', 'LONG', 'LONG TOTALE', 
                        'LONG TABLIER', 'LARG HORS TOUT', 'LARG CARROSSABLE', 
                        'SUPERF TABLIER', 'AN CONST', 'STATUT', 'DJMA', 
                        '% CAMIONS', 'NBRE VOIES']

    # all interventions are assumed to be 1 year long
    schema_interventions = ['NO STRUCT', 'ANNEE DEBUT TRAVAUX', 'CODE ACT', 
                            'ELEMENT', 'IDENTIFIANT ELEMENT']

    schema_inspectors = ['IDENTIFIANT INSPECTION', 'IDENTIFIANT INSPECTEUR']

    return schema_inspections, schema_structures, schema_interventions, schema_inspectors