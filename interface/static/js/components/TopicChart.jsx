import React from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ErrorBar } from "recharts";

const TopicChart = () => {
  const [chartData, setChartData] = React.useState(null);

  React.useEffect(() => {
    if (window.topicData) {
      const transformedData = Array.isArray(window.topicData)
        ? window.topicData
        : Object.keys(window.topicData)
            .filter((key) => !isNaN(parseInt(key)))
            .map((key) => window.topicData[key]);
      console.log("Transformed data:", transformedData);
      setChartData(transformedData);
    }
  }, []);

  if (!chartData) return null;

  const tooltipStyle = {
    backgroundColor: "white",
    padding: "16px",
    boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)",
    border: "1px solid #e5e7eb",
    borderRadius: "8px",
  };

  const tooltipTitleStyle = {
    fontSize: "18px",
    fontWeight: "bold",
    marginBottom: "4px",
  };

  const tooltipTextStyle = {
    color: "#374151",
    marginBottom: "4px",
  };

  const tooltipSubtextStyle = {
    color: "#6B7280",
    fontSize: "14px",
  };

  return (
    <div className="topic-container">
      <div style={{ width: "100%", height: "400px" }}>
        <ResponsiveContainer>
          <BarChart data={chartData} margin={{ top: 20, right: 70, left: 50, bottom: 60 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="category"
              label={{
                value: "Topics",
                position: "bottom",
                style: { fontWeight: "bold" },
                dy: 16,
              }}
            />
            <YAxis
              domain={[0, 100]}
              label={{
                value: "Average Percentage Focused",
                angle: -90,
                position: "insideLeft",
                style: { fontWeight: "bold" },
                dy: 100,
                dx: -10,
              }}
            />
            <Tooltip
              cursor={{ fill: 'rgba(0, 53, 107, 0.1)' }} 
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div style={tooltipStyle}>
                      <p style={tooltipTitleStyle}>{label}</p>
                      <p style={tooltipTextStyle}>Average Focus: {data.focusPercentage.toFixed(1)}%</p>
                      <p style={tooltipSubtextStyle}>Standard Deviation: ±{data.standardDeviation.toFixed(1)}%</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="focusPercentage" fill="#00356B">
              <ErrorBar dataKey="standardDeviation" width={4} strokeWidth={2} stroke="#666" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TopicChart;