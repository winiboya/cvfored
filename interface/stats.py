import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy
from datetime import datetime
import random

class Analytics:
    """
    Class for outputting analytics.
    """
    def __init__(self, input_file, topic_names=None, topic_starts=None, topic_ends=None, colors=None):
        """
        Initializes analytics with the given path to a csv file.
        """
        self.input_file = input_file
        self.topic_names = topic_names
        self.topic_starts = topic_starts
        self.topic_ends = topic_ends
    
    def table(self):
        df = pd.read_csv(self.input_file)
        table = df.groupby(["Frame Number", "Prediction"]).size() / df.groupby('Frame Number').size() * 100
        table = table.rename("percentages")
        table = table.unstack(fill_value=0)
        table = table.stack().reset_index(name='Percentages')
        table_df = table.reset_index()
        subset = table_df[table_df['Prediction'] == "focused"]
        #print(subset)
        return subset
        
    def line_chart(self):

        df = self.table()
        # temp = df['Frame Number']
        x = df['Frame Number'].values
        # temp = df['Percentages']
        y = df['Percentages'].values

        fig = go.Figure(
            data=go.Scatter(
                x=x, 
                y=y, 
                mode='lines+markers',
                marker=dict(size=8, color='#00356B', symbol='circle'),
                hoverinfo='x+y',
                line=dict(color='#00356B', width=2),
                hovertemplate='<b>Percent Focused</b>: %{y:.2f}<br><b>Timestamp</b>: %{x}<extra></extra>'
            )
        )

        fig.update_layout(
            title_x=0.5,
            xaxis=dict(
                title='<b>Timestamp</b>',
                showgrid=True,
                gridcolor='#D4D5D9',
                griddash='dash', 
                gridwidth=1,
                zeroline=False,
                showline=False,
                showticklabels=True,
                title_font=dict(size=20, color='black'),
                tickfont=dict(size=14, color='#D4D5D9'),
                tickangle=0
            ),
            yaxis=dict(
                title='<b>Percentage Focused</b>',
                showgrid=True,
                gridcolor='#D4D5D9',
                griddash='dash', 
                gridwidth=1,
                zeroline=False,
                showline=False,
                title_font=dict(size=20, color='black', weight='bold'),
                tickfont=dict(size=14, color='#D4D5D9'),
                range=[0, 100]
            ),
            plot_bgcolor='white',
            margin=dict(l=40, r=40, t=40, b=40),
            font=dict(family="Inter, sans-serif", size=12)
        )

        if self.topic_names is not None:
            shapes = []
            colors = []
            vertical_lines = []
            for i in range(len(self.topic_names)):
                colors.append(self.generate_random_color())
            for i in range(len(self.topic_names)):
                # shapes.append(dict(
                #         type='line',
                #         x0=self.topic_starts[i], x1=self.topic_starts[i],
                #         y0=0, y1=100,
                #         line=dict(color=colors[i], dash='dot', width=2)
                #     ))
                # shapes.append(dict(
                #         type='line',
                #         x0=self.topic_ends[i], x1=self.topic_ends[i],
                #         y0=0, y1=100,
                #         line=dict(color=colors[i], dash='dot', width=2)
                #     ))
                # fig.update_layout(
                # shapes=shapes)
                vertical_lines.append({"x": [self.topic_starts[i], self.topic_starts[i]], "y": [0, 100], "hovertext": f"{self.topic_names[i]} starts", "color": colors[i]})
                vertical_lines.append({"x": [self.topic_ends[i], self.topic_ends[i]], "y": [0, 100], "hovertext": f"{self.topic_names[i]} ends", "color": colors[i]})
            for line in vertical_lines:
                fig.add_trace(
                    go.Scatter(
                        x=line["x"],
                        y=line["y"],
                        mode='lines',
                        line=dict(color=line["color"], width=2, dash='dot'),  # Dotted style
                        hoverinfo='text',  # Enable hover text
                        hovertext=line["hovertext"],  # Text to display on hover
                    )
                )
            fig.update_layout(showlegend=False)

            self.colors=colors

        return fig
    
    def get_average(self):
        df = pd.read_csv(self.input_file)
        if df.empty:
            return 0
        df = self.table()
        return round(df['Percentages'].mean())
    
    def get_max(self):
        df = pd.read_csv(self.input_file)
        if df.empty:
            return 0
        df = self.table()
        return df['Percentages'].max()
    
    def get_min(self):
        df = pd.read_csv(self.input_file)
        if df.empty:
            return 0
        df = self.table()
        return df['Percentages'].min()
    
    def get_student_count(self):
        df = pd.read_csv(self.input_file)
        if df.empty:
            return 0
        df['Face Number'] = df['Face Number'].apply(lambda x: int(x[4:]))
        max_count = df['Face Number'].max()
        return max_count
    
    def get_average_student_count(self):
        df = pd.read_csv(self.input_file)
        if df.empty:
            return 0
        df['Face Number'] = df['Face Number'].apply(lambda x: int(x[4:]))
        average = df['Face Number'].mean()
        return round(average)

    def get_minutes(self):
        df = self.table()
        if df.empty:
            return 0
        end = df['Frame Number'].max()
        start = df['Frame Number'].min()
        time_format = "%M:%S"
        start_obj = datetime.strptime(start, time_format)
        end_obj = datetime.strptime(end, time_format)
        diff = end_obj - start_obj

        minutes = diff.seconds // 60
        seconds = diff.seconds % 60

        return minutes
        
    def get_std(self):
        df = self.table()
        if df.empty:
            return 0
        std = df['Percentages'].std()
        return std

    def stats(self):

        line_chart_fig = self.line_chart()

        average = self.get_average()

        max = self.get_max()

        min = self.get_min()

        student_count = self.get_student_count()
        
        average_student_count = self.get_average_student_count()

        minutes = self.get_minutes()

        std = self.get_std()
        
        table_fig = go.Figure(data=[go.Table(header=dict(values=['', ''], height=30),
                 cells=dict(values=[
            ["Average Attentiveness", "Maximum Attentiveness", "Minimum Attentiveness", "Student Count", "Average Student Count", "Minutes Analyzed", "Standard Deviation"],
            [f"{average}%", f"{max}%", f"{min}%", f"{student_count}", f"{average_student_count}", f"{minutes}", f"{std}"]
        ], font=dict(size=16), height=35))])

        # Add title
        table_fig.update_layout(title="Summary Statistics", font=dict(size=18), title_x=0.5, title_y=0.88)

        all_stats = {
            'line_chart': self.line_chart(),
            'table': table_fig,
            'average': average,
            'max': max,
            'min': min, 
            'student_count': student_count, 
            'average_student_count': average_student_count,
            'minutes': minutes,
            'std': std
        }

        return all_stats
    
    def topic_separation(self):
    
        output_csvs = []

        for i in range(len(self.topic_names)):
            df = pd.read_csv(self.input_file)
            df = df[df['Frame Number'] >= self.topic_starts[i]]
            df = df[df['Frame Number'] <= self.topic_ends[i]]

            output_path = f"{self.topic_names[i]}.csv"
            df.to_csv(output_path, index=False)
            output_csvs.append(output_path)
        
        return output_csvs
    
    def generate_random_color(self):
        return f"#{random.randint(0, 0xFFFFFF):06x}"
    
    def topic_results(self):

        topics = dict()
        averages, average_student_counts, mins, std = [], [], [], []
        output_csvs = self.topic_separation()
        for file in output_csvs:
            analyze_topic = Analytics(file)
            all_stats = analyze_topic.stats()

            # add to topic dictionary
            topics[file[:-4]] = all_stats

            # add to topic-wide stats
            averages.append(all_stats['average'])
            average_student_counts.append(all_stats['average_student_count'])
            mins.append(all_stats['minutes'])
            std.append(all_stats['std'])

        # colors = []
        # for i in range(len(self.topic_names)):
        #     colors.append(self.generate_random_color())
        # print(colors)

        # create topic-wide figures
        averages_fig = go.Figure(data=[go.Bar(x=self.topic_names, y=averages, marker=dict(
                color=self.colors
            ), error_y=dict(
            type='data',
            array=std, 
            visible=True 
        ))])
        
        averages_fig.update_layout(
        title={
            'text': "Topic Comparison",
            'x': 0.5,  # Center the title
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis=dict(
            title='Topics',
            titlefont=dict(size=16),
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            title='Average Percentage Focused',
            titlefont=dict(size=16),
            tickfont=dict(size=14),
            range=[0, 100]  # Set y-axis range
        ),
        plot_bgcolor='white',  # Background color
        paper_bgcolor='#F9F9F9',  # Overall background color
        margin=dict(l=50, r=50, t=50, b=50),  # Margins
        font=dict(family='Arial, sans-serif', size=12)  # Default font
    )

        average_student_count_fig = go.Figure(data=[go.Bar(x=self.topic_names, y=average_student_counts)])

        mins_fig = go.Figure(data=[go.Bar(x=self.topic_names, y=mins)])

        return averages_fig, average_student_count_fig, mins_fig, topics

