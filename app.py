from flask import Flask, render_template, request, url_for
import math
from scipy.stats import norm
import requests
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# ============================
# Option Strike Calculator Code (unchanged)
# ============================


def calculate_d2(S, K, r, sigma, T):
    try:
        d2 = (math.log(S / K) + (r - 0.5 * sigma**2) * T) / \
            (sigma * math.sqrt(T))
    except Exception as e:
        d2 = None
    return d2


def calculate_risk_neutral_probability(d2):
    if d2 is not None:
        return norm.cdf(d2)
    return None


@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    error = None
    if request.method == 'POST':
        try:
            S = float(request.form.get('S', 100))
            r = float(request.form.get('r', 0.01))
            sigma = float(request.form.get('sigma', 0.20))
            T = float(request.form.get('T', 0.25))
            strikes_input = request.form.get('strikes', '')
            if strikes_input:
                strike_prices = [float(s.strip())
                                 for s in strikes_input.split(',')]
            else:
                strike_prices = [95, 100, 105]

            for K in strike_prices:
                d2 = calculate_d2(S, K, r, sigma, T)
                prob = calculate_risk_neutral_probability(d2)
                results.append({
                    'strike': K,
                    'd2': round(d2, 3) if d2 is not None else 'Error',
                    'probability': round(prob, 3) if prob is not None else 'Error'
                })
        except Exception as e:
            error = f"An error occurred: {e}"

    return render_template('index.html', results=results, error=error)

# ============================
# Weather Search Code
# ============================


def get_coordinates(location):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location, "format": "json"}
    headers = {"User-Agent": "MyWeatherApp/1.0 (your_email@example.com)"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise Exception(
            f"Nominatim error: {response.status_code} {response.text}")
    data = response.json()
    if data:
        lat = data[0]["lat"]
        lon = data[0]["lon"]
        return float(lat), float(lon)
    return None, None


def get_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "current_weather": "true"}
    response = requests.get(url, params=params)
    return response.json()


@app.route('/weather', methods=['GET', 'POST'])
def weather():
    weather_data = None
    error = None
    location = ""
    if request.method == 'POST':
        location = request.form.get('location')
        lat, lon = get_coordinates(location)
        if lat is None or lon is None:
            error = "Location not found. Please try a different location."
        else:
            weather_data = get_weather(lat, lon)
    return render_template('weather.html', weather=weather_data, error=error, location=location)

# ============================
# Weather Strike Calculator with Plot
# ============================


def generate_plot(current_temp, strike_temp, sigma):
    """
    Generates a plot of the normal probability density function centered at current_temp,
    shades the area to the right of the strike_temp, and returns the plot as a base64 string.
    """
    # Create a range of temperature values around the current temperature
    x = np.linspace(current_temp - 4*sigma, current_temp + 4*sigma, 500)
    y = norm.pdf(x, current_temp, sigma)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Temperature PDF')

    # Shade the area to the right of the strike temperature
    x_fill = np.linspace(strike_temp, current_temp + 4*sigma, 500)
    ax.fill_between(x_fill, norm.pdf(x_fill, current_temp, sigma),
                    color='gray', alpha=0.5, label='Exceedance Area')

    ax.axvline(strike_temp, color='red', linestyle='--',
               label=f'Strike: {strike_temp} °C')
    ax.set_title('Probability of Temperature Exceeding Strike')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Probability Density')
    ax.legend()

    # Save the plot to a BytesIO buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close(fig)
    return image_base64


@app.route('/weather_strike', methods=['GET', 'POST'])
def weather_strike():
    result = None
    error = None
    location = ""
    plot_url = None
    if request.method == 'POST':
        try:
            location = request.form.get('location')
            strike_temp = float(request.form.get('strike_temp'))
            lat, lon = get_coordinates(location)
            if lat is None or lon is None:
                error = "Location not found. Please try a different location."
            else:
                weather_data = get_weather(lat, lon)
                if weather_data and "current_weather" in weather_data:
                    current_temp = weather_data["current_weather"]["temperature"]
                    # Assume a fixed volatility (e.g., 5 °C) for demonstration purposes
                    sigma = 5.0
                    # Calculate probability that temperature exceeds strike
                    prob = 1 - norm.cdf((strike_temp - current_temp) / sigma)
                    result = {
                        "location": location,
                        "current_temp": current_temp,
                        "strike_temp": strike_temp,
                        "probability": round(prob, 3)
                    }
                    # Generate the probability plot
                    plot_url = generate_plot(current_temp, strike_temp, sigma)
                else:
                    error = "Weather data not available."
        except Exception as e:
            error = f"An error occurred: {e}"
    return render_template('weather_strike.html', result=result, error=error, location=location, plot_url=plot_url)


if __name__ == '__main__':
    # Run on port 5010 as requested.
    app.run(debug=True, port=5010)
