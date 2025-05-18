# app.py - Main Flask Application
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import matplotlib
import json
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)


# Load the data
def load_data():
    """Load cricket data from cricsheet.org PSL dataset"""
    try:
        # Load the processed PSL data file
        df = pd.read_csv('all_matches.csv')
        logger.info(f"Successfully loaded data with {len(df)} rows.")
        return df
    except FileNotFoundError:
        logger.error("Error: all_matches.csv file not found.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None

# Set plot style and size
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('colorblind')



# Helper function to get cached data and stats
def get_data_and_stats():
    """Get cached data and stats or recompute if needed"""
    if not hasattr(get_data_and_stats, 'data'):
        logger.info("Initializing data and stats for the first time")
        df = load_data()
        if df is not None:
            df, bowler_stats, batter_stats, matchup_stats = preprocess_data(df)
            if df is not None:
                get_data_and_stats.data = (df, bowler_stats, batter_stats, matchup_stats)
            else:
                logger.error("Preprocessing failed")
                return None, {}, {}, {}
        else:
            logger.error("Failed to load data")
            return None, {}, {}, {}

    return get_data_and_stats.data
# Data preprocessing
def preprocess_data(df):
    """Preprocess the cricket data for modeling"""
    if df is None or df.empty:
        logger.error("Cannot preprocess empty dataframe")
        return None, {}, {}, {}

    try:
        logger.info("Starting data preprocessing...")

        # Create matchup identifier
        df['matchup'] = df['bowler'] + '_vs_' + df['striker']

        # Create outcome columns
        df['is_wicket'] = df['wicket_type'].apply(lambda x: 0 if pd.isna(x) or x == '' else 1)
        df['is_boundary'] = df['runs_off_bat'].apply(lambda x: 1 if x in [4, 6] else 0)
        df['is_dot_ball'] = ((df['runs_off_bat'] == 0) & (df['extras'] == 0)).astype(int)

        # Create features for the model
        # Calculate historical stats for each player
        bowler_stats = {}
        batter_stats = {}

        logger.info("Calculating bowler statistics...")
        for bowler in df['bowler'].dropna().unique():
            bowler_df = df[df['bowler'] == bowler]
            bowler_stats[bowler] = {
                'wickets': bowler_df['is_wicket'].sum(),
                'balls_bowled': len(bowler_df),
                'runs_conceded': bowler_df['runs_off_bat'].sum() + bowler_df['extras'].sum(),
                'boundaries_conceded': bowler_df['is_boundary'].sum(),
                'dot_balls': bowler_df['is_dot_ball'].sum()
            }

        logger.info("Calculating batter statistics...")
        for batter in df['striker'].dropna().unique():
            batter_df = df[df['striker'] == batter]
            batter_stats[batter] = {
                'runs_scored': batter_df['runs_off_bat'].sum(),
                'balls_faced': len(batter_df),
                'boundaries_hit': batter_df['is_boundary'].sum(),
                'times_dismissed': batter_df['is_wicket'].sum(),
                'dot_balls_faced': batter_df['is_dot_ball'].sum()
            }

        # Add historical stats to each row
        logger.info("Adding player statistics to dataframe...")
        df['bowler_wicket_rate'] = df['bowler'].apply(
            lambda x: bowler_stats[x]['wickets'] / max(1, bowler_stats[x]['balls_bowled']))
        df['bowler_economy'] = df['bowler'].apply(
            lambda x: (bowler_stats[x]['runs_conceded'] / max(1, bowler_stats[x]['balls_bowled'])) * 6)
        df['bowler_boundary_rate'] = df['bowler'].apply(
            lambda x: bowler_stats[x]['boundaries_conceded'] / max(1, bowler_stats[x]['balls_bowled']))
        df['bowler_dot_rate'] = df['bowler'].apply(
            lambda x: bowler_stats[x]['dot_balls'] / max(1, bowler_stats[x]['balls_bowled']))

        df['batter_strike_rate'] = df['striker'].apply(
            lambda x: (batter_stats[x]['runs_scored'] / max(1, batter_stats[x]['balls_faced'])) * 100)
        df['batter_boundary_rate'] = df['striker'].apply(
            lambda x: batter_stats[x]['boundaries_hit'] / max(1, batter_stats[x]['balls_faced']))
        df['batter_dismissal_rate'] = df['striker'].apply(
            lambda x: batter_stats[x]['times_dismissed'] / max(1, batter_stats[x]['balls_faced']))
        df['batter_dot_rate'] = df['striker'].apply(
            lambda x: batter_stats[x]['dot_balls_faced'] / max(1, batter_stats[x]['balls_faced']))

        # Calculate head-to-head stats
        logger.info("Calculating matchup statistics...")
        matchup_stats = {}

        for matchup in df['matchup'].unique():
            matchup_df = df[df['matchup'] == matchup]

            try:
                bowler, batter = matchup.split('_vs_')
                total_balls = len(matchup_df)

                if total_balls > 0:
                    wickets = matchup_df['is_wicket'].sum()
                    runs = matchup_df['runs_off_bat'].sum()
                    boundaries = matchup_df['is_boundary'].sum()
                    dot_balls = matchup_df['is_dot_ball'].sum()

                    matchup_stats[matchup] = {
                        'total_balls': total_balls,
                        'wickets': wickets,
                        'runs': runs,
                        'wicket_rate': wickets / total_balls,
                        'runs_per_ball': runs / total_balls,
                        'boundary_rate': boundaries / total_balls,
                        'dot_rate': dot_balls / total_balls
                    }
            except Exception as e:
                logger.warning(f"Error processing matchup {matchup}: {str(e)}")

        logger.info("Preprocessing completed successfully.")
        return df, bowler_stats, batter_stats, matchup_stats

    except Exception as e:
        logger.error(f"Error in preprocessing data: {str(e)}")
        return None, {}, {}, {}


# Model training
def train_models(df):
    """Train prediction models for different outcomes"""
    try:
        logger.info("Training prediction models...")

        # Features for prediction
        features = [
            'bowler_wicket_rate', 'bowler_economy', 'bowler_boundary_rate', 'bowler_dot_rate',
            'batter_strike_rate', 'batter_boundary_rate', 'batter_dismissal_rate', 'batter_dot_rate'
        ]

        # Train wicket prediction model
        X_wicket = df[features].fillna(0)
        y_wicket = df['is_wicket']
        wicket_model = RandomForestClassifier(n_estimators=100, random_state=42)
        wicket_model.fit(X_wicket, y_wicket)

        # Train boundary prediction model
        X_boundary = df[features].fillna(0)
        y_boundary = df['is_boundary']
        boundary_model = RandomForestClassifier(n_estimators=100, random_state=42)
        boundary_model.fit(X_boundary, y_boundary)

        # Train dot ball prediction model
        X_dot = df[features].fillna(0)
        y_dot = df['is_dot_ball']
        dot_model = RandomForestClassifier(n_estimators=100, random_state=42)
        dot_model.fit(X_dot, y_dot)

        # Save models
        if not os.path.exists('models'):
            os.makedirs('models')

        with open('models/wicket_model.pkl', 'wb') as f:
            pickle.dump(wicket_model, f)

        with open('models/boundary_model.pkl', 'wb') as f:
            pickle.dump(boundary_model, f)

        with open('models/dot_model.pkl', 'wb') as f:
            pickle.dump(dot_model, f)

        logger.info("Models trained and saved successfully.")
        return wicket_model, boundary_model, dot_model

    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        return None, None, None


# Prediction functions
def predict_matchup(bowler, batter, wicket_model, boundary_model, dot_model, df, bowler_stats, batter_stats):
    """Predict outcomes for a specific bowler-batter matchup"""
    try:
        logger.info(f"Making prediction for {bowler} vs {batter}")

        # Validate inputs
        if bowler not in bowler_stats:
            raise ValueError(f"Bowler '{bowler}' not found in dataset")

        if batter not in batter_stats:
            raise ValueError(f"Batter '{batter}' not found in dataset")

        # Prepare input features for prediction
        features = {
            'bowler_wicket_rate': bowler_stats[bowler]['wickets'] / max(1, bowler_stats[bowler]['balls_bowled']),
            'bowler_economy': (bowler_stats[bowler]['runs_conceded'] / max(1,
                                                                           bowler_stats[bowler]['balls_bowled'])) * 6,
            'bowler_boundary_rate': bowler_stats[bowler]['boundaries_conceded'] / max(1, bowler_stats[bowler][
                'balls_bowled']),
            'bowler_dot_rate': bowler_stats[bowler]['dot_balls'] / max(1, bowler_stats[bowler]['balls_bowled']),
            'batter_strike_rate': (batter_stats[batter]['runs_scored'] / max(1, batter_stats[batter][
                'balls_faced'])) * 100,
            'batter_boundary_rate': batter_stats[batter]['boundaries_hit'] / max(1,
                                                                                 batter_stats[batter]['balls_faced']),
            'batter_dismissal_rate': batter_stats[batter]['times_dismissed'] / max(1,
                                                                                   batter_stats[batter]['balls_faced']),
            'batter_dot_rate': batter_stats[batter]['dot_balls_faced'] / max(1, batter_stats[batter]['balls_faced'])
        }

        features_df = pd.DataFrame([features])

        # Make predictions
        wicket_prob = wicket_model.predict_proba(features_df)[0][1]
        boundary_prob = boundary_model.predict_proba(features_df)[0][1]
        dot_prob = dot_model.predict_proba(features_df)[0][1]

        # Historical head-to-head stats
        matchup = f"{bowler}_vs_{batter}"
        head_to_head = {}

        matchup_df = df[df['matchup'] == matchup]
        if len(matchup_df) > 0:
            # Convert NumPy types to Python native types
            head_to_head = {
                'total_balls': int(len(matchup_df)),
                'wickets': int(matchup_df['is_wicket'].sum()),
                'runs': int(matchup_df['runs_off_bat'].sum()),
                'boundaries': int(matchup_df['is_boundary'].sum()),
                'dot_balls': int(matchup_df['is_dot_ball'].sum())
            }

        prediction_result = {
            'wicket_probability': round(float(wicket_prob * 100), 2),
            'boundary_probability': round(float(boundary_prob * 100), 2),
            'dot_ball_probability': round(float(dot_prob * 100), 2),
            'head_to_head': head_to_head
        }

        logger.info(
            f"Prediction complete: wicket={prediction_result['wicket_probability']}%, boundary={prediction_result['boundary_probability']}%, dot={prediction_result['dot_ball_probability']}%")
        return prediction_result

    except Exception as e:
        logger.error(f"Error predicting matchup: {str(e)}")
        raise

# Routes
@app.route('/')
def index():
    """Render the main dashboard page"""
    try:
        # Get list of all batters and bowlers for dropdowns
        df = load_data()

        # Make sure the dataframe has the necessary columns
        if df is not None and not df.empty and 'striker' in df.columns and 'bowler' in df.columns:
            # Extract unique player names, ensuring no None/NaN values
            batters = sorted([str(x) for x in df['striker'].dropna().unique() if x])
            bowlers = sorted([str(x) for x in df['bowler'].dropna().unique() if x])

            logger.info(f"Found {len(batters)} batters and {len(bowlers)} bowlers")
            logger.info(f"Sample batters: {batters[:5] if len(batters) >= 5 else batters}")
            logger.info(f"Sample bowlers: {bowlers[:5] if len(bowlers) >= 5 else bowlers}")
        else:
            # Error handling
            logger.error("Invalid or empty dataset")
            error_msg = "Error: Could not load valid cricket data. Please ensure the dataset is properly formatted."
            return render_template('error.html', error=error_msg)

        return render_template('index.html', batters=batters, bowlers=bowlers)

    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        error_msg = "An unexpected error occurred while loading the dashboard. Please check the server logs."
        return render_template('error.html', error=error_msg)


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for match-up prediction"""
    try:
        # Get form data
        bowler = request.form.get('bowler')
        batter = request.form.get('batter')

        logger.info(f"Prediction request for bowler: {bowler}, batter: {batter}")

        # Validate input
        if not bowler or not batter:
            logger.warning("Missing bowler or batter parameter")
            return jsonify({'error': 'Missing bowler or batter parameter'}), 400

        # Get data and models
        try:
            df, bowler_stats, batter_stats, _ = get_data_and_stats()
            if df is None:
                logger.error("Failed to get data and stats")
                return jsonify({'error': 'Failed to load cricket data'}), 500
        except Exception as e:
            logger.error(f"Error getting data and stats: {str(e)}")
            return jsonify({'error': 'Error loading data: ' + str(e)}), 500

        try:
            wicket_model, boundary_model, dot_model = get_models()
            if None in (wicket_model, boundary_model, dot_model):
                logger.error("One or more models failed to load")
                return jsonify({'error': 'Failed to load prediction models'}), 500
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return jsonify({'error': 'Error loading prediction models: ' + str(e)}), 500

        # Check if bowler and batter exist in our dataset
        if bowler not in bowler_stats:
            logger.warning(f"Bowler {bowler} not found in dataset")
            return jsonify({'error': f'Bowler {bowler} not found in dataset'}), 404

        if batter not in batter_stats:
            logger.warning(f"Batter {batter} not found in dataset")
            return jsonify({'error': f'Batter {batter} not found in dataset'}), 404

        # Make prediction
        try:
            prediction = predict_matchup(bowler, batter, wicket_model, boundary_model, dot_model, df, bowler_stats,
                                         batter_stats)
            return jsonify(prediction)
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({'error': 'Error generating prediction: ' + str(e)}), 500

    except Exception as e:
        logger.error(f"Unexpected error in predict route: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500


@app.route('/heatmap')
def heatmap():
    """Generate and display the match-up heatmap"""
    try:
        # Get data
        try:
            df, _, _, _ = get_data_and_stats()
            if df is None:
                logger.error("Failed to get data for heatmap - df is None")
                error_msg = "Error: Could not load cricket data for heatmap"
                return render_template('error.html', error=error_msg)
        except Exception as e:
            logger.error(f"Error loading data for heatmap: {str(e)}")
            error_msg = f"Error loading data: {str(e)}"
            return render_template('error.html', error=error_msg)

        # Get top players by frequency
        try:
            top_batters = df['striker'].value_counts().head(10).index.tolist()
            top_bowlers = df['bowler'].value_counts().head(10).index.tolist()

            if not top_batters or not top_bowlers:
                logger.error(f"No top players found - batters: {len(top_batters)}, bowlers: {len(top_bowlers)}")
                error_msg = "Error: Not enough player data found for heatmap"
                return render_template('error.html', error=error_msg)

            logger.info(f"Found {len(top_batters)} top batters and {len(top_bowlers)} top bowlers for heatmap")
        except Exception as e:
            logger.error(f"Error getting top players: {str(e)}")
            error_msg = f"Error processing player data: {str(e)}"
            return render_template('error.html', error=error_msg)

        # Generate heatmap
        try:
            heatmap_img = generate_heatmap(df, top_batters, top_bowlers)
            if heatmap_img is None:
                logger.error("Failed to generate heatmap - result is None")
                # Instead of returning an error page, render the heatmap template with a warning
                return render_template('heatmap.html', heatmap_img=None)
        except Exception as e:
            logger.error(f"Error generating heatmap: {str(e)}")
            # Instead of returning an error page, render the heatmap template with a warning
            return render_template('heatmap.html', heatmap_img=None)

        # Return the heatmap page
        return render_template('heatmap.html', heatmap_img=heatmap_img)

    except Exception as e:
        logger.error(f"Unexpected error in heatmap route: {str(e)}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        return render_template('error.html', error=error_msg)


# Generate visualization for matchups
def generate_heatmap(df, top_batters, top_bowlers):
    """Generate a heatmap of wicket probabilities for top batters and bowlers"""
    try:
        # Create a matrix of wicket probabilities
        wicket_matrix = pd.DataFrame(index=top_bowlers, columns=top_batters)

        for bowler in top_bowlers:
            for batter in top_batters:
                matchup = f"{bowler}_vs_{batter}"
                matchup_df = df[df['matchup'] == matchup]

                if len(matchup_df) > 5:  # Minimum sample size
                    wicket_prob = matchup_df['is_wicket'].mean() * 100
                else:
                    wicket_prob = 0  # Use 0 instead of np.nan

                wicket_matrix.loc[bowler, batter] = wicket_prob

        # Convert all values to float to ensure compatibility
        wicket_matrix = wicket_matrix.astype(float)

        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(wicket_matrix, annot=True, cmap='Blues', fmt='.1f', cbar_kws={'label': 'Wicket Probability (%)'})
        plt.title('Bowler-Batter Matchup - Wicket Probability')
        plt.tight_layout()

        # Save to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        # Encode to base64
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"Error in generate_heatmap: {str(e)}")
        return None


@app.route('/simulate', methods=['GET', 'POST'])
def simulate():
    """Scenario simulation page"""
    try:
        if request.method == 'GET':
            try:
                df, _, _, _ = get_data_and_stats()
                if df is None:
                    logger.error("Failed to get data for simulation page - df is None")
                    error_msg = "Error: Could not load cricket data for simulation"
                    return render_template('error.html', error=error_msg)

                # Extract player lists
                batters = sorted([str(x) for x in df['striker'].dropna().unique() if x])
                bowlers = sorted([str(x) for x in df['bowler'].dropna().unique() if x])

                # Extract venue list
                venues = sorted([str(x) for x in df['venue'].dropna().unique() if x])

                if not batters or not bowlers:
                    logger.error(f"No players found - batters: {len(batters)}, bowlers: {len(bowlers)}")
                    error_msg = "Error: No player data found for simulation"
                    return render_template('error.html', error=error_msg)

                logger.info(
                    f"Simulation page loaded with {len(batters)} batters, {len(bowlers)} bowlers, and {len(venues)} venues")
                return render_template('simulate.html', batters=batters, bowlers=bowlers, venues=venues)
            except Exception as e:
                logger.error(f"Error loading simulation page: {str(e)}")
                error_msg = f"Error loading simulation page: {str(e)}"
                return render_template('error.html', error=error_msg)
        else:
            # Handle simulation form submission
            try:
                # Get form data
                bowling_team = request.form.get('bowling_team', '')
                batting_team = request.form.get('batting_team', '')
                venue = request.form.get('venue', '')
                bowlers = request.form.getlist('bowlers')
                batters = request.form.getlist('batters')

                try:
                    overs = int(request.form.get('overs', 5))
                except ValueError:
                    logger.warning("Invalid overs value, defaulting to 5")
                    overs = 5

                # Log received data for debugging
                logger.info(
                    f"Simulation request: {batting_team} vs {bowling_team} at {venue}, {len(batters)} batters, {len(bowlers)} bowlers, {overs} overs")

                # Validate scenario data
                if not bowlers:
                    logger.warning("No bowlers selected for simulation")
                    error_msg = "Please select at least one bowler for simulation"
                    return render_template('error.html', error=error_msg)

                if not batters:
                    logger.warning("No batters selected for simulation")
                    error_msg = "Please select at least one batter for simulation"
                    return render_template('error.html', error=error_msg)

                # Create scenario
                scenario = {
                    'bowling_team': bowling_team,
                    'batting_team': batting_team,
                    'venue': venue,
                    'bowlers': bowlers,
                    'batters': batters,
                    'overs': overs
                }

                # Run simulation
                try:
                    simulation_results = run_simulation(scenario)
                    if simulation_results is None:
                        logger.error("Simulation failed to return results")
                        error_msg = "Error: Simulation failed to produce results"
                        return render_template('error.html', error=error_msg)
                except Exception as e:
                    logger.error(f"Error running simulation: {str(e)}")
                    error_msg = f"Error running simulation: {str(e)}"
                    return render_template('error.html', error=error_msg)

                # Convert any NumPy values to regular Python types for JSON serialization
                try:
                    # Convert top-level stats
                    for key in ['runs', 'wickets', 'balls', 'dot_balls', 'boundaries']:
                        if key in simulation_results and hasattr(simulation_results[key], 'item'):
                            simulation_results[key] = simulation_results[key].item()

                    # Convert run_rate
                    if 'run_rate' in simulation_results and hasattr(simulation_results['run_rate'], 'item'):
                        simulation_results['run_rate'] = simulation_results['run_rate'].item()

                    # Convert bowler stats
                    for bowler in simulation_results.get('bowler_stats', {}):
                        for stat in ['overs', 'runs', 'wickets', 'economy']:
                            if stat in simulation_results['bowler_stats'][bowler] and hasattr(
                                    simulation_results['bowler_stats'][bowler][stat], 'item'):
                                simulation_results['bowler_stats'][bowler][stat] = \
                                    simulation_results['bowler_stats'][bowler][stat].item()

                    # Convert batter stats
                    for batter in simulation_results.get('batter_stats', {}):
                        for stat in ['runs', 'balls', 'strike_rate']:
                            if stat in simulation_results['batter_stats'][batter] and hasattr(
                                    simulation_results['batter_stats'][batter][stat], 'item'):
                                simulation_results['batter_stats'][batter][stat] = \
                                    simulation_results['batter_stats'][batter][stat].item()
                except Exception as e:
                    logger.error(f"Error converting NumPy types: {str(e)}")
                    # Continue anyway, we'll handle any serialization errors elsewhere

                # Add venue information to results
                simulation_results['venue'] = venue

                # Render results template
                try:
                    return render_template('simulation_results.html', results=simulation_results)
                except Exception as e:
                    logger.error(f"Error rendering simulation results: {str(e)}")
                    error_msg = f"Error displaying simulation results: {str(e)}"
                    return render_template('error.html', error=error_msg)

            except Exception as e:
                logger.error(f"Error processing simulation form: {str(e)}")
                error_msg = f"Error running simulation: {str(e)}"
                return render_template('error.html', error=error_msg)

    except Exception as e:
        logger.error(f"Unexpected error in simulate route: {str(e)}")
        error_msg = f"An unexpected error occurred: {str(e)}"
        return render_template('error.html', error=error_msg)

def get_models():
    """Get cached models or retrain if needed"""
    if not hasattr(get_models, 'models'):
        models_path = 'models'
        if os.path.exists(models_path) and all(os.path.exists(f"{models_path}/{model_name}.pkl") for model_name in
                                               ['wicket_model', 'boundary_model', 'dot_model']):
            # Load models from files
            try:
                logger.info("Loading models from files")
                with open(f"{models_path}/wicket_model.pkl", 'rb') as f:
                    wicket_model = pickle.load(f)

                with open(f"{models_path}/boundary_model.pkl", 'rb') as f:
                    boundary_model = pickle.load(f)

                with open(f"{models_path}/dot_model.pkl", 'rb') as f:
                    dot_model = pickle.load(f)

                get_models.models = (wicket_model, boundary_model, dot_model)
            except Exception as e:
                logger.error(f"Error loading models from files: {str(e)}")
                return None, None, None
        else:
            # Train models
            logger.info("Models not found, training new models")
            df, _, _, _ = get_data_and_stats()
            if df is not None:
                wicket_model, boundary_model, dot_model = train_models(df)
                if None not in (wicket_model, boundary_model, dot_model):
                    get_models.models = (wicket_model, boundary_model, dot_model)
                else:
                    logger.error("Failed to train one or more models")
                    return None, None, None
            else:
                logger.error("No data available for training models")
                return None, None, None

    return get_models.models


def run_simulation(scenario):
    """Run a simulation based on the given scenario"""
    try:
        logger.info(
            f"Running simulation with {len(scenario['bowlers'])} bowlers and {len(scenario['batters'])} batters")

        df, bowler_stats, batter_stats, _ = get_data_and_stats()
        wicket_model, boundary_model, dot_model = get_models()

        # Fix this condition - check each component separately
        if df is None or wicket_model is None or boundary_model is None or dot_model is None:
            logger.error("Cannot run simulation: missing data or models")
            return None

        # Setup simulation
        bowling_team = scenario['bowling_team']
        batting_team = scenario['batting_team']
        venue = scenario.get('venue', '')
        bowlers = scenario['bowlers']
        batters = scenario['batters']
        overs = scenario['overs']

        # Calculate venue adjustment factors
        venue_factors = {
            'batting_factor': 1.0,  # default - no adjustment
            'bowling_factor': 1.0  # default - no adjustment
        }

        if venue:
            try:
                # Get venue-specific data
                venue_df = df[df['venue'] == venue]

                if len(venue_df) > 0:
                    # Calculate average runs per ball at this venue
                    venue_runs_per_ball = venue_df['runs_off_bat'].mean()

                    # Calculate average runs per ball across all venues
                    all_venues_runs_per_ball = df['runs_off_bat'].mean()

                    if all_venues_runs_per_ball > 0:
                        # Calculate batting adjustment factor
                        venue_factors['batting_factor'] = venue_runs_per_ball / all_venues_runs_per_ball

                    # Calculate wicket probability at this venue
                    venue_wicket_prob = venue_df['is_wicket'].mean()

                    # Calculate wicket probability across all venues
                    all_venues_wicket_prob = df['is_wicket'].mean()

                    if all_venues_wicket_prob > 0:
                        # Calculate bowling adjustment factor
                        venue_factors['bowling_factor'] = venue_wicket_prob / all_venues_wicket_prob

                    logger.info(
                        f"Venue factors for {venue}: batting={venue_factors['batting_factor']:.2f}, bowling={venue_factors['bowling_factor']:.2f}")
            except Exception as e:
                logger.error(f"Error calculating venue factors: {str(e)}")
                # Continue with default factors

        # Validate all players exist in the dataset
        for bowler in bowlers:
            if bowler not in bowler_stats:
                logger.warning(f"Bowler {bowler} not found in dataset")
                raise ValueError(f"Bowler {bowler} not found in dataset")

        for batter in batters:
            if batter not in batter_stats:
                logger.warning(f"Batter {batter} not found in dataset")
                raise ValueError(f"Batter {batter} not found in dataset")

        # Initialize results
        results = {
            'runs': 0,
            'wickets': 0,
            'balls': 0,
            'dot_balls': 0,
            'boundaries': 0,
            'bowler_stats': {},
            'batter_stats': {},
            'venue': venue
        }

        # Initialize player stats
        for bowler in bowlers:
            results['bowler_stats'][bowler] = {
                'overs': 0,
                'runs': 0,
                'wickets': 0,
                'economy': 0
            }

        for batter in batters:
            results['batter_stats'][batter] = {
                'runs': 0,
                'balls': 0,
                'strike_rate': 0,
                'dismissal': False
            }

        # Run simulation ball by ball
        current_batter_idx = 0
        current_bowler_idx = 0

        for over in range(overs):
            # Select bowler for this over
            current_bowler = bowlers[current_bowler_idx % len(bowlers)]
            current_bowler_idx += 1

            for ball in range(6):
                # Check if all batters are out
                active_batters = [b for b in batters if not results['batter_stats'][b]['dismissal']]
                if len(active_batters) == 0:
                    logger.info("Simulation ended: all batters out")
                    break

                # Select current batter
                current_batter = active_batters[current_batter_idx % len(active_batters)]

                # Predict outcome
                prediction = predict_matchup(current_bowler, current_batter, wicket_model, boundary_model, dot_model,
                                             df, bowler_stats, batter_stats)

                # Apply venue adjustments to probabilities
                wicket_chance = prediction['wicket_probability'] / 100 * venue_factors['bowling_factor']
                boundary_chance = prediction['boundary_probability'] / 100 * venue_factors['batting_factor']
                dot_chance = prediction['dot_ball_probability'] / 100 / venue_factors['batting_factor']

                # Ensure probabilities stay within reasonable bounds
                wicket_chance = min(max(wicket_chance, 0.01), 0.5)
                boundary_chance = min(max(boundary_chance, 0.05), 0.4)
                dot_chance = min(max(dot_chance, 0.1), 0.7)

                # Generate random number to determine outcome
                outcome = np.random.random()

                if outcome < wicket_chance:
                    # Wicket
                    results['wickets'] += 1
                    results['balls'] += 1
                    results['bowler_stats'][current_bowler]['wickets'] += 1
                    results['batter_stats'][current_batter]['balls'] += 1
                    results['batter_stats'][current_batter]['dismissal'] = True
                    current_batter_idx += 1
                    logger.debug(f"Ball {over}.{ball + 1}: WICKET! {current_bowler} dismissed {current_batter}")
                elif outcome < wicket_chance + boundary_chance:
                    # Boundary (4 or 6)
                    boundary_val = 4 if np.random.random() < 0.7 else 6
                    results['runs'] += boundary_val
                    results['balls'] += 1
                    results['boundaries'] += 1
                    results['bowler_stats'][current_bowler]['runs'] += boundary_val
                    results['batter_stats'][current_batter]['runs'] += boundary_val
                    results['batter_stats'][current_batter]['balls'] += 1
                    logger.debug(
                        f"Ball {over}.{ball + 1}: BOUNDARY! {current_batter} hit {boundary_val} off {current_bowler}")
                elif outcome < wicket_chance + boundary_chance + dot_chance:
                    # Dot ball
                    results['balls'] += 1
                    results['dot_balls'] += 1
                    results['batter_stats'][current_batter]['balls'] += 1
                    logger.debug(f"Ball {over}.{ball + 1}: DOT BALL. {current_batter} faced {current_bowler}")
                else:
                    # Regular runs (1, 2, or 3)
                    runs_val = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                    results['runs'] += runs_val
                    results['balls'] += 1
                    results['bowler_stats'][current_bowler]['runs'] += runs_val
                    results['batter_stats'][current_batter]['runs'] += runs_val
                    results['batter_stats'][current_batter]['balls'] += 1
                    logger.debug(
                        f"Ball {over}.{ball + 1}: {runs_val} RUNS. {current_batter} scored off {current_bowler}")

                    # Switch striker if odd number of runs
                    if runs_val % 2 == 1:
                        current_batter_idx += 1

            # Switch striker at end of over
            current_batter_idx += 1

            # Update bowler overs
            results['bowler_stats'][current_bowler]['overs'] += 1

        # Calculate final stats
        logger.info("Calculating final simulation stats")
        for bowler in bowlers:
            overs = results['bowler_stats'][bowler]['overs']
            runs = results['bowler_stats'][bowler]['runs']
            if overs > 0:
                results['bowler_stats'][bowler]['economy'] = runs / overs

        for batter in batters:
            runs = results['batter_stats'][batter]['runs']
            balls = results['batter_stats'][batter]['balls']
            if balls > 0:
                results['batter_stats'][batter]['strike_rate'] = (runs / balls) * 100

        # Final match summary
        results['overs'] = f"{results['balls'] // 6}.{results['balls'] % 6}"
        results['run_rate'] = results['runs'] / max(1, results['balls'] / 6)

        logger.info(
            f"Simulation complete: {results['runs']}/{results['wickets']} in {results['overs']} overs at {venue}")
        return results

    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        return None

# Create error template
def create_error_template():
    """Create error template file if it doesn't exist"""
    if not os.path.exists('templates'):
        os.makedirs('templates')

    error_template_path = 'templates/error.html'
    if not os.path.exists(error_template_path):
        with open(error_template_path, 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Cricket Player Match-Up Prediction Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
    <style>
        .header-bg {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
        }
    </style>
</head>
<body>
    <header class="header-bg py-4 mb-4">
        <div class="container">
            <h1 class="text-center">Cricket Player Match-Up Prediction Dashboard</h1>
        </div>
    </header>

    <div class="container">
        <div class="row">
            <div class="col-md-12">
                <div class="card shadow">
                    <div class="card-header bg-danger text-white">
                        <h2 class="h4 mb-0">Error</h2>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-danger" role="alert">
                            {{ error }}
                        </div>
                        <p>Please try again or contact the administrator if the problem persists.</p>
                        <a href="/" class="btn btn-primary">Return to Dashboard</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-white py-4 mt-5">
        <div class="container text-center">
            <p>&copy; 2025 Cricket Analytics Dashboard</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
</body>
</html>
            ''')


def create_templates():
    """Create template files for the application if they don't exist"""
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Dictionary of template names and their content
    templates = {
        'error.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Cricket Player Match-Up Prediction Dashboard</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
    <style>
        .header-bg {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
        }
    </style>
</head>
<body>
    <header class="header-bg py-4 mb-4">
        <div class="container">
            <h1 class="text-center">Cricket Player Match-Up Prediction Dashboard</h1>
        </div>
    </header>

    <div class="container">
        <div class="row">
            <div class="col-md-12">
                <div class="card shadow">
                    <div class="card-header bg-danger text-white">
                        <h2 class="h4 mb-0">Error</h2>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-danger" role="alert">
                            {{ error }}
                        </div>
                        <p>Please try again or contact the administrator if the problem persists.</p>
                        <a href="/" class="btn btn-primary">Return to Dashboard</a>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-white py-4 mt-5">
        <div class="container text-center">
            <p>&copy; 2025 Cricket Analytics Dashboard</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        ''',
        'heatmap.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Match-Up Heatmap</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
    <style>
        .header-bg {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
        }
    </style>
</head>
<body>
    <header class="header-bg py-4 mb-4">
        <div class="container">
            <h1 class="text-center">Cricket Player Match-Up Heatmap</h1>
        </div>
    </header>

    <div class="container">
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card shadow">
                    <div class="card-header bg-info text-white">
                        <h2 class="h4 mb-0">Wicket Probability Heatmap</h2>
                    </div>
                    <div class="card-body text-center">
                        <p class="mb-4">This heatmap shows wicket probabilities for top bowlers vs. top batters</p>
                        {% if heatmap_img %}
                            <img src="{{ heatmap_img }}" class="img-fluid" alt="Match-up Heatmap">
                        {% else %}
                            <div class="alert alert-warning">
                                No heatmap data available. There may not be enough match data.
                            </div>
                        {% endif %}
                        <p class="mt-4 text-muted">Darker blue indicates higher wicket probability</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row">
            <div class="col-md-12">
                <a href="/" class="btn btn-secondary mb-4">Back to Dashboard</a>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-white py-4 mt-5">
        <div class="container text-center">
            <p>&copy; 2025 Cricket Analytics Dashboard</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        ''',
        'simulate.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Scenario Simulation</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/select2/4.1.0-rc.0/css/select2.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/select2-bootstrap-5-theme/1.3.0/select2-bootstrap-5-theme.min.css">
    <style>
        .header-bg {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
        }
        .dropdown-alert {
            display: none;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <header class="header-bg py-4 mb-4">
        <div class="container">
            <h1 class="text-center">Cricket Scenario Simulation</h1>
        </div>
    </header>

    <div class="container">
        <!-- Alert for when dropdowns are empty -->
        <div id="emptyDropdownAlert" class="alert alert-warning dropdown-alert" role="alert">
            <strong>Data Loading...</strong> Please wait a moment while player data is being loaded. If this message persists, try returning to the dashboard and coming back.
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card shadow">
                    <div class="card-header bg-warning text-dark">
                        <h2 class="h4 mb-0">Configure Simulation</h2>
                    </div>
                    <div class="card-body">
                        <form action="/simulate" method="post" id="simulationForm">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="batting_team" class="form-label">Batting Team</label>
                                    <input type="text" class="form-control" id="batting_team" name="batting_team" required>
                                </div>
                                <div class="col-md-6">
                                    <label for="bowling_team" class="form-label">Bowling Team</label>
                                    <input type="text" class="form-control" id="bowling_team" name="bowling_team" required>
                                </div>
                            </div>

                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="batters" class="form-label">Select Batters</label>
                                    <select class="form-select batters-select" id="batters" name="batters" multiple required>
                                        {% for batter in batters %}
                                        <option value="{{ batter }}">{{ batter }}</option>
                                        {% endfor %}
                                    </select>
                                    <small class="text-muted">Select at least 4 batters</small>
                                </div>
                                <div class="col-md-6">
                                    <label for="bowlers" class="form-label">Select Bowlers</label>
                                    <select class="form-select bowlers-select" id="bowlers" name="bowlers" multiple required>
                                        {% for bowler in bowlers %}
                                        <option value="{{ bowler }}">{{ bowler }}</option>
                                        {% endfor %}
                                    </select>
                                    <small class="text-muted">Select at least 2 bowlers</small>
                                </div>
                            </div>

                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="overs" class="form-label">Number of Overs</label>
                                    <input type="number" class="form-control" id="overs" name="overs" min="1" max="20" value="5" required>
                                </div>
                            </div>

                            <div class="d-grid gap-2 d-md-flex justify-content-md-end">
                                <a href="/" class="btn btn-secondary me-md-2">Back to Dashboard</a>
                                <button type="submit" class="btn btn-warning" id="simulateBtn">Run Simulation</button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-white py-4 mt-5">
        <div class="container text-center">
            <p>&copy; 2025 Cricket Analytics Dashboard</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/select2/4.1.0-rc.0/js/select2.min.js"></script>
    <script>
        $(document).ready(function() {
            // Check if dropdowns have options
            const batterOptions = $('#batters option').length;
            const bowlerOptions = $('#bowlers option').length;

            // Log for debugging
            console.log("Batters in dropdown:", batterOptions);
            console.log("Bowlers in dropdown:", bowlerOptions);

            // If either dropdown is empty, show the alert and disable the form
            if (batterOptions === 0 || bowlerOptions === 0) {
                $('#emptyDropdownAlert').show();
                $('#simulateBtn').prop('disabled', true);

                // Try to reload the page after 5 seconds to get fresh data
                setTimeout(function() {
                    location.reload();
                }, 5000);
            } else {
                // Initialize Select2 when we have data
                $('.batters-select').select2({
                    theme: 'bootstrap-5',
                    placeholder: 'Select batters',
                    maximumSelectionLength: 11
                });

                $('.bowlers-select').select2({
                    theme: 'bootstrap-5',
                    placeholder: 'Select bowlers',
                    maximumSelectionLength: 5
                });
            }

            // Form validation
            $('#simulationForm').on('submit', function(e) {
                const selectedBatters = $('#batters').val();
                const selectedBowlers = $('#bowlers').val();

                if (!selectedBatters || selectedBatters.length < 4) {
                    e.preventDefault();
                    alert('Please select at least 4 batters');
                    return false;
                }

                if (!selectedBowlers || selectedBowlers.length < 2) {
                    e.preventDefault();
                    alert('Please select at least 2 bowlers');
                    return false;
                }
            });
        });
    </script>
</body>
</html>
        ''',
        'simulation_results.html': '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cricket Simulation Results</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
    <style>
        .header-bg {
            background: linear-gradient(135deg, #3498db, #2c3e50);
            color: white;
        }
        .result-card {
            transition: all 0.3s;
        }
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <header class="header-bg py-4 mb-4">
        <div class="container">
            <h1 class="text-center">Cricket Simulation Results</h1>
        </div>
    </header>

    <div class="container">
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="card shadow">
                    <div class="card-header bg-success text-white">
                        <h2 class="h4 mb-0">Match Summary</h2>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="card result-card text-center mb-3">
                                    <div class="card-header bg-primary text-white">
                                        Score
                                    </div>
                                    <div class="card-body">
                                        <h3 class="display-5">{{ results.runs | default(0) }} / {{ results.wickets | default(0) }}</h3>
                                        <p>{{ results.overs | default('0.0') }} overs</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card result-card text-center mb-3">
                                    <div class="card-header bg-info text-white">
                                        Run Rate
                                    </div>
                                    <div class="card-body">
                                        <h3 class="display-5">{{ '%0.2f' | format(results.run_rate | default(0)) }}</h3>
                                        <p>runs per over</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card result-card text-center mb-3">
                                    <div class="card-header bg-warning text-dark">
                                        Boundaries
                                    </div>
                                    <div class="card-body">
                                        <h3 class="display-5">{{ results.boundaries | default(0) }}</h3>
                                        <p>4s and 6s</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card result-card text-center mb-3">
                                    <div class="card-header bg-secondary text-white">
                                        Dot Balls
                                    </div>
                                    <div class="card-body">
                                        <h3 class="display-5">{{ results.dot_balls | default(0) }}</h3>
                                        <p>{{ '%0.1f' | format((results.dot_balls / results.balls * 100) if results.balls else 0) }}% of deliveries</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-6">
                <div class="card shadow">
                    <div class="card-header bg-primary text-white">
                        <h2 class="h4 mb-0">Batters Performance</h2>
                    </div>
                    <div class="card-body">
                        {% if results.batter_stats %}
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Batter</th>
                                    <th>Runs</th>
                                    <th>Balls</th>
                                    <th>Strike Rate</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for batter, stats in results.batter_stats.items() %}
                                <tr>
                                    <td>{{ batter }}</td>
                                    <td>{{ stats.runs | default(0) }}</td>
                                    <td>{{ stats.balls | default(0) }}</td>
                                    <td>{{ '%0.2f' | format(stats.strike_rate | default(0)) }}</td>
                                    <td>{{ "Out" if stats.dismissal else "Not Out" }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                        {% else %}
                        <div class="alert alert-warning">No batter data available</div>
                        {% endif %}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card shadow">
                    <div class="card-header bg-danger text-white">
                        <h2 class="h4 mb-0">Bowlers Performance</h2>
                    </div>
                    <div class="card-body">
                        {% if results.bowler_stats %}
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Bowler</th>
                                    <th>Overs</th>
                                    <th>Runs</th>
                                    <th>Wickets</th>
                                    <th>Economy</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for bowler, stats in results.bowler_stats.items() %}
                                <tr>
                                    <td>{{ bowler }}</td>
                                    <td>{{ stats.overs | default(0) }}</td>
                                    <td>{{ stats.runs | default(0) }}</td>
                                    <td>{{ stats.wickets | default(0) }}</td>
                                    <td>{{ '%0.2f' | format(stats.economy | default(0)) }}</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                        {% else %}
                        <div class="alert alert-warning">No bowler data available</div>
                        {% endif %}
                    </div>
                </div>
            </div>
        </div>

        <div class="row mb-4">
            <div class="col-md-12">
                <div class="d-grid gap-2 d-md-flex">
                    <a href="/simulate" class="btn btn-warning me-md-2">Run Another Simulation</a>
                    <a href="/" class="btn btn-secondary">Back to Dashboard</a>
                </div>
            </div>
        </div>
    </div>

    <footer class="bg-dark text-white py-4 mt-5">
        <div class="container text-center">
            <p>&copy; 2025 Cricket Analytics Dashboard</p>
        </div>
    </footer>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        '''
    }

    # Create each template file if it doesn't exist
    for template_name, template_content in templates.items():
        template_path = os.path.join('templates', template_name)
        if not os.path.exists(template_path):
            with open(template_path, 'w') as f:
                f.write(template_content)
            print(f"Created template: {template_name}")


if __name__ == '__main__':
    # Ensure templates exist
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create templates if needed
    create_templates()

    # Initialize data and models on startup
    try:
        df, _, _, _ = get_data_and_stats()
        if df is not None:
            logger.info("Initial data loading successful")
        else:
            logger.error("Failed to load initial data")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")

    # Run the app
    app.run(debug=True)