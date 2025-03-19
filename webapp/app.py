
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import ipaddress
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import joblib


scaler = joblib.load("./data/scaler.pkl")
pca = joblib.load("./data/pca.pkl")


expected_columns = [
    "Timestamp", "Source IP Address", "Destination IP Address", "Source Port",
    "Destination Port", "Protocol", "Packet Length", "Packet Type", "Traffic Type",
    "Payload Data", "Malware Indicators", "Anomaly Scores", "Alerts/Warnings",
    "Attack Type", "Attack Signature", "Action Taken", "Severity Level",
    "User Information", "Device Information", "Network Segment", "Geo-location Data",
    "Proxy Information", "Firewall Logs", "IDS/IPS Alerts", "Log Source"
]
input = pd.read_csv('./data/sample_samples.csv')
def extract_device_info(user_agent):
    """
    Extracts Operating System (OS), Browser, and Device Type from a user-agent string.
    """
    user_agent = str(user_agent).lower()

    if "windows" in user_agent:
        os = "Windows"
    elif "mac os" in user_agent or "macintosh" in user_agent:
        os = "MacOS"
    elif "linux" in user_agent:
        os = "Linux"
    elif "android" in user_agent:
        os = "Android"
    elif "iphone" in user_agent or "ipad" in user_agent or "ios" in user_agent:
        os = "iOS"
    else:
        os = "Other"

    if "chrome" in user_agent:
        browser = "Chrome"
    elif "firefox" in user_agent:
        browser = "Firefox"
    elif "safari" in user_agent and "chrome" not in user_agent:
        browser = "Safari"
    elif "msie" in user_agent or "trident" in user_agent:
        browser = "Internet Explorer"
    elif "edge" in user_agent:
        browser = "Edge"
    else:
        browser = "Other"

    if "mobile" in user_agent or "android" in user_agent or "iphone" in user_agent:
        device_type = "Mobile"
    else:
        device_type = "Desktop"

    return pd.Series([os, browser, device_type])

def extract_ip_features(ip):
    try:
        ip_obj = ipaddress.ip_address(ip)
        octets = str(ip).split('.')
        octet_1 = int(octets[0])
        octet_2 = int(octets[1])
        octet_3 = int(octets[2])
        octet_4 = int(octets[3])
        
    except:
        octet_1, octet_2, octet_3, octet_4 = 0, 0, 0, 0

    return pd.Series([octet_1, octet_2, octet_3, octet_4])

def time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"
    
def one_hot_encode_with_defaults(df, column_name, expected_categories):
    dummies = pd.get_dummies(df[column_name], prefix='', prefix_sep='')
    for category in expected_categories:
        if category not in dummies.columns:
            dummies[category] = 0
    dummies = dummies.astype(int)
    df = df.drop(columns=[column_name], errors='ignore')
    df = pd.concat([df, dummies], axis=1)
    return df

def preprocess_samples(df):
    if isinstance(df, list):
        df = pd.DataFrame(df)
    
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a Pandas DataFrame.")
    if len(df.columns) < len(expected_columns):
        print(f"Auto-assigning headers to match expected columns")
        df.columns = expected_columns[:len(df.columns)]
    print(f"DEBUG: Original DataFrame Shape: {df.shape}")
    print(df.head())
    df.columns = expected_columns[:len(df.columns)]

    missing_features = [col for col in expected_columns if col not in df.columns]
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
        for col in missing_cols:
            df[col] = 0 
    if missing_features:
        print(f"Warning: Missing features detected: {missing_features}")
        for col in missing_features:
            df[col]

    df.fillna(0, inplace=True)  

    columns = [
    "User Information",
    "Geo-location Data",
    "Payload Data"
    ]
    df = df.drop(columns=columns)
    missing_cols = ['Malware Indicators', 'Alerts/Warnings', 'Firewall Logs', 'IDS/IPS Alerts', 'Proxy Information']
    df[missing_cols] = df[missing_cols].fillna(0)
    df['Malware Indicators'] = df['Malware Indicators'].apply(lambda x: 1 if x == 'IoC Detected' else 0)
    df['Alerts/Warnings'] = df['Alerts/Warnings'].apply(lambda x: 1 if x == 'Alert Triggered' else 0)
    df['Firewall Logs'] = df['Firewall Logs'].apply(lambda x: 1 if x == 'Log Data' else 0)
    df['IDS/IPS Alerts'] = df['IDS/IPS Alerts'].apply(lambda x: 1 if x == 'Alert Data' else 0)
    df['Packet Type'] = df['Packet Type'].apply(lambda x: 1 if x == 'Data' else 0)
    df['Log Source'] = df['Log Source'].apply(lambda x: 1 if x == 'Server' else 0)
    df['Attack Signature'] = df['Attack Signature'].apply(lambda x: 1 if x == 'Known Pattern A' else 0)

    df = one_hot_encode_with_defaults(df, 'Protocol', ['ICMP', 'UDP'])
    df = one_hot_encode_with_defaults(df, 'Traffic Type', ['HTTP', 'DNS'])
    df = one_hot_encode_with_defaults(df, 'Action Taken', ['Logged', 'Blocked'])
    df = one_hot_encode_with_defaults(df, 'Severity Level', ['Low', 'Medium'])
    df = one_hot_encode_with_defaults(df, 'Network Segment', ['Segment A', 'Segment B'])

    if "Device Information" in df.columns:
        device_info = df["Device Information"].apply(lambda x: pd.Series(extract_device_info(x)))

        device_info.columns = ["OS", "Browser", "Device Type"]

        df = pd.concat([df, device_info], axis=1)
    else:
        df["OS"] = "Unknown"
        df["Browser"] = "Unknown"
        df["Device Type"] = "Unknown"

    for col in ["OS", "Browser", "Device Type"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df = pd.get_dummies(df, columns=["OS", "Browser", "Device Type"], drop_first=True, dtype=int)

    df['Src_Port_Binned'] = pd.cut(df['Source Port'], bins=[0, 1023, 49151, 65535], labels=['Well-Known', 'Registered', 'Dynamic'])
    df['Dst_Port_Binned'] = pd.cut(df['Destination Port'], bins=[0, 1023, 49151, 65535], labels=['Well-Known', 'Registered', 'Dynamic'])

    df['Log_Packet_Length'] = np.log1p(df['Packet Length'])
    
    if 'Source IP Address' in df.columns and 'Destination IP Address' in df.columns:
        df[['Src_Octet1', 'Src_Octet2', 'Src_Octet3', 'Src_Octet4']] = df['Source IP Address'].str.split('.', expand=True).astype(int)
        df[['Dst_Octet1', 'Dst_Octet2', 'Dst_Octet3', 'Dst_Octet4']] = df['Destination IP Address'].str.split('.', expand=True).astype(int)

    df['Src_Octet1*Dst_Octet1'] = df['Src_Octet1'] * df['Dst_Octet1']
    df['Src_Octet2*Dst_Octet2'] = df['Src_Octet2'] * df['Dst_Octet2']
    df['Src_Octet3*Dst_Octet3'] = df['Src_Octet3'] * df['Dst_Octet3']
    df['Src_Octet4*Dst_Octet4'] = df['Src_Octet4'] * df['Dst_Octet4']

    df = pd.get_dummies(df, columns=['Src_Port_Binned', 'Dst_Port_Binned'], drop_first=True)
    df = df.drop(columns=['Source IP Address', 'Destination IP Address', 'Proxy Information']) 

    if "Proxy Information" not in df.columns:
        df["Proxy Information"] = 0

    df['Proxy_Used'] = df['Proxy Information'].apply(lambda x: 0 if x == 0 or pd.isnull(x) else 1)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df["Year"] = df["Timestamp"].dt.year
    df = pd.get_dummies(df, columns=['Year'], drop_first=True, dtype=int)
    df["Month"] = df["Timestamp"].dt.month
    df["Day"] = df["Timestamp"].dt.day
    df["Hour"] = df["Timestamp"].dt.hour
    df["Minute"] = df["Timestamp"].dt.minute
    df["Second"] = df["Timestamp"].dt.second
    df["Day_of_Week"] = df["Timestamp"].dt.dayofweek

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')

    df['Hour'] = df['Timestamp'].dt.hour
    df['Day_of_Week'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month

    df['Time_of_Day'] = df['Hour'].apply(time_of_day)

    df = pd.get_dummies(df, columns=['Day_of_Week', 'Month', 'Time_of_Day'], drop_first=True)

    df = df.drop(columns = 'Timestamp')

    attack_mapping = {'Malware': 0, 'Intrusion': 1, 'DDoS': 2}
    df['Attack Type'] = df['Attack Type'].map(attack_mapping)
    attack_type = df["Attack Type"] if "Attack Type" in df.columns else None
    bool_columns = df.select_dtypes(include=['bool']).columns
    df[bool_columns] = df[bool_columns].astype(int)
    numerical_columns = df.select_dtypes(include=['number']).columns
    numerical_columns = numerical_columns.drop("Attack Type", errors="ignore")
    
    y = df["Attack Type"] if "Attack Type" in df.columns else None

    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    df = df.drop(columns=["Attack Type"], errors="ignore")
    pca_features = pca.feature_names_in_
    for col in pca_features:
        if col not in df.columns:
            df[col] = 0
    
    df = df[pca_features]
    transformed_data = pca.transform(df)
    pca_columns = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
    transformed_df = pd.DataFrame(transformed_data, columns=pca_columns)
    top_20_pcs = [
    "PC9", "PC41", "PC36", "PC7", "PC37", "PC6", "PC39", "PC48", "PC8", "PC11",
    "PC5", "PC44", "PC40", "PC43", "PC45", "PC4", "PC38", "PC35", "PC55", "PC49"
    ]
    transformed_df = transformed_df[top_20_pcs]
    if y is not None:
        transformed_df["Attack Type"] = y.values

    transformed_df.to_csv("Processed_Samples.csv", index=False)


    print(df.head())
    return pd.DataFrame(transformed_df)

preprocess_samples(input)
