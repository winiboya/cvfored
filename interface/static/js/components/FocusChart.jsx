import React from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const FocusChart = () => {
	const [focusData, setFocusData] = React.useState([]);

	React.useEffect(() => {
		// The data is already in the correct format
		if (window.focusData) {
			setFocusData(window.focusData);
		}
	}, []);

	const formatTime = (timeStr) => {
		const [minutes, seconds] = timeStr.split(":").map(Number);
		return `${minutes} minutes ${seconds} seconds`;
	};

	const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
          return (
            <div style={{
              backgroundColor: '#ffffff',
              padding: '16px',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
              borderRadius: '8px',
              border: '1px solid #e5e7eb',
              minWidth: '200px',
              textAlign: 'left'
            }}>
              <div style={{ 
                fontSize: '20px',
                fontWeight: 'bold',
                marginBottom: '8px',
                color: '#000000'
              }}>
                {`${payload[0].value}% Focused`}
              </div>
              <div style={{ 
                color: '#9CA3AF', 
                fontSize: '14px'
              }}>
                {formatTime(label)}
              </div>
            </div>
          );
        }
        return null;
      };
    


	return (
		<div className="focus-container">
			<div style={{ width: "100%", height: "400px" }}>
				<ResponsiveContainer>
					<LineChart data={focusData} margin={{ top: 20, right: 70, left: 50, bottom: 60 }}>
						<CartesianGrid strokeDasharray="3 3" />
						<XAxis
							dataKey="timestamp"
							label={{ value: "Time", position: "bottom", style: { fontWeight: "bold" }, offset: 0, dx:-30, fill: "black" }}
							tick={false}
                            dy={16}
						/>
						<YAxis
							domain={[0, 100]}
							label={{ value: "Percentage Focused", angle: -90, position: "insideLeft", style: { fontWeight: "bold" }, textAnchor: "middle", dy: 100, dx: -10, fill: "black" }}
							tick={{ fill: "#666" }}
                            dx={-10}
						/>
						<Tooltip content={<CustomTooltip />} />
						<Line type="monotone" dataKey="focusPercentage" stroke="#00356B" strokeWidth={2} dot={{ fill: "#00356B", size: 8 }} />
					</LineChart>
				</ResponsiveContainer>
			</div>
		</div>
	);
};

export default FocusChart;
