import csv
def get_patient_names(image_list_file):
    patient_names = {}
    with open(image_list_file, "r") as f:
        csvReader = csv.reader(f)
        next(csvReader, None)
        for line in csvReader:
            patient_name = line[0].split("/")[2]
            if patient_name in patient_names:
                patient_names[patient_name] += 1
            else:
                patient_names[patient_name] = 1
    return patient_names