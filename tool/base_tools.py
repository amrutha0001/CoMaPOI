import pandas as pd
from agentscope.message import Msg
from agentscope.service import (ServiceToolkit, ServiceResponse, ServiceExecStatus)

def load_user_history(history_path):
    """Load user's historical trajectory data from train_sample_processed.csv file."""
    USER_HISTORY = {}
    #print("Starting to load user historical trajectory data...")

    try:
        # Read CSV file
        df = pd.read_csv(history_path, parse_dates=['time'])

        # Convert day_of_week to integer (if needed)
        df['day_of_week'] = df['day_of_week'].astype(int)

        # Group by user_id
        grouped = df.groupby('user_id')

        for user_id, group in grouped:
            # Store each user's data as DataFrame
            USER_HISTORY[user_id] = group.reset_index(drop=True)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error loading historical trajectory data: {e}")

    #print(f"Historical trajectory data loading complete, loaded data for {len(USER_HISTORY)} users.")
    return USER_HISTORY

def time_distribution_summary(user_data):
    """Generate a summary of user's time preferences."""
    # Ensure 'time' column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(user_data['time']):
        user_data['time'] = pd.to_datetime(user_data['time'])

    # Extract hour information
    user_data['hour'] = user_data['time'].dt.hour

    # Count frequency of visits for each hour
    hour_counts = user_data['hour'].value_counts().sort_index()
    hour_counts = hour_counts.sort_values(ascending=False)
    
    # Prepare explanation information
    time_explanation = [
        f"Time: {hour:02d}:00-{(hour+1)%24:02d}:00, Frequency: {count}"
        for hour, count in hour_counts.items()
    ]

    return time_explanation

def day_distribution_summary(user_data):
    """Generate a summary of user's day of week preferences."""
    # Ensure day_of_week is integer
    user_data['day_of_week'] = user_data['day_of_week'].astype(int)

    # Count frequency of visits for each day of week
    day_counts = user_data['day_of_week'].value_counts().sort_index()
    
    # Map day numbers to day names
    day_names = {
        1: "Monday",
        2: "Tuesday",
        3: "Wednesday",
        4: "Thursday",
        5: "Friday",
        6: "Saturday",
        7: "Sunday"
    }
    
    # Prepare explanation information
    day_explanation = [
        f"Day: {day_names[day]}, Frequency: {count}"
        for day, count in day_counts.items()
    ]

    return day_explanation

def category_distribution_summary(user_data):
    """Generate a summary of user's POI category preferences."""
    # Count frequency of visits for each category
    category_counts = user_data['category'].value_counts()
    
    # Prepare explanation information
    category_explanation = [
        f"Category: {category}, Frequency: {count}"
        for category, count in category_counts.items()
    ]

    return category_explanation

def poi_distribution_summary(user_data):
    """Generate a summary of user's POI preferences."""
    # Count frequency of visits for each POI
    poi_counts = user_data['poi_id'].value_counts()
    
    # Prepare explanation information
    poi_explanation = [
        f"POI ID: {poi_id}, Frequency: {count}"
        for poi_id, count in poi_counts.items()
    ]

    return poi_explanation

def get_all_information_tool(user_id, data):
    """
    Tool to get all information about a user's historical trajectory.
    
    Args:
        user_id: User ID
        data: Dataset name
        
    Returns:
        ServiceResponse: Response containing user information
    """
    try:
        # Load user history
        history_path = f'dataset_all/{data}/train/{data}_train_sample.csv'
        USER_HISTORY = load_user_history(history_path)
        
        # Check if user exists
        if int(user_id) not in USER_HISTORY:
            return ServiceResponse(
                status=ServiceExecStatus.FAILED,
                content=f"User ID {user_id} not found in dataset {data}."
            )
        
        # Get user data
        user_data = USER_HISTORY[int(user_id)]
        
        # Generate summaries
        time_summary = time_distribution_summary(user_data)
        day_summary = day_distribution_summary(user_data)
        category_summary = category_distribution_summary(user_data)
        poi_summary = poi_distribution_summary(user_data)
        
        # Prepare response
        response = {
            "user_id": user_id,
            "dataset": data,
            "time_distribution": time_summary,
            "day_distribution": day_summary,
            "category_distribution": category_summary,
            "poi_distribution": poi_summary,
            "trajectory_length": len(user_data),
            "trajectory_data": user_data.to_dict(orient='records')
        }
        
        return ServiceResponse(
            status=ServiceExecStatus.SUCCESS,
            content=response
        )
    
    except Exception as e:
        return ServiceResponse(
            status=ServiceExecStatus.FAILED,
            content=f"Error getting user information: {e}"
        )
