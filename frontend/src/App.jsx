import { useState } from "react";
import "./app.css";

function App() {
  const [form, setForm] = useState({
    from_city: "",
    to_city: "",
    travel_date: "",
    travel_time: "12:00",
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await fetch("http://127.0.0.1:5000/predict_route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      alert("Failed to fetch prediction. Make sure backend is running.");
    }
  };

  const severityClass = (level) => {
    switch (level) {
      case 1: return "safe";
      case 2: return "moderate";
      case 3: return "danger";
      case 4: return "extreme";
      default: return "unknown";
    }
  };

  return (
    <div className="app">
      <h1>Citizen Safety Assistant</h1>

      <div className="dashboard-grid">
        {/* Left side: form + result */}
        <div className="left-panel">
          <form onSubmit={handleSubmit}>
            <input
              type="text"
              name="from_city"
              placeholder="From City (e.g., Miami, US)"
              value={form.from_city}
              onChange={handleChange}
              required
            />
            <input
              type="text"
              name="to_city"
              placeholder="To City (e.g., Orlando, US)"
              value={form.to_city}
              onChange={handleChange}
              required
            />
            <div className="date-time">
              <input
                type="date"
                name="travel_date"
                value={form.travel_date}
                onChange={handleChange}
                required
              />
              <input
                type="time"
                name="travel_time"
                value={form.travel_time}
                onChange={handleChange}
              />
            </div>
            <button type="submit">Predict Safety</button>
          </form>

          {result && (
            <div className={`result ${severityClass(result.severity)}`}>
              <h2>{result.message}</h2>
              <p><strong>Route:</strong> {result.route}</p>
              <p><strong>Distance:</strong> {result.distance_miles} miles</p>
              <p><strong>Duration:</strong> {result.estimated_duration_min} min</p>
              <p><strong>Weather from:</strong> {result.weather_from.weather_main}, {result.weather_from.weather_desc}</p>
              <p><strong>Weather to:</strong> {result.weather_to.weather_main}, {result.weather_to.weather_desc}</p>
            </div>
          )}
        </div>

        {/* Right side: heatmap */}
        <div className="right-panel">
          <div className="heatmap-container">
            <h2>Accident Heatmap</h2>
            <iframe
              src="/public/Accident_Hotspots.html"
              title="Accident Heatmap"
            ></iframe>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
