import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Analytics:
    """
    Class for outputting analytics.
    """
    def __init__(self, input_file):
        """
        Initializes analytics with the given path to a csv file.
        """
        self.input_file = input_file
    
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
        # fig.show()

        return fig
    
    def get_average(self):
        df = pd.read_csv(self.input_file)
        if df.empty:
            return 0
        df = self.table()
        return df['Percentages'].mean()
    
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

    def get_minutes(self):
        df = self.table()
        if df.empty:
            return 0
        max_timestamp = df['Frame Number'].max()
        if int(max_timestamp[:2]) == 0:
            return "< 1"
        else:
            return f"{int(max_timestamp[:2])}"

    def stats(self):

        average = self.get_average()

        max = self.get_max()

        min = self.get_min()

        student_count = self.get_student_count()

        minutes = self.get_minutes()
        
        fig = go.Figure(data=[go.Table(header=dict(values=['', ''], height=30),
                 cells=dict(values=[
            ["Average Attentiveness", "Maximum Attentiveness", "Minimum Attentiveness", "Student Count", "Minutes Analyzed"],
            [f"{average}%", f"{max}%", f"{min}%", f"{student_count}", f"{minutes}"]
        ], font=dict(size=16), height=35))])

        # Add title
        fig.update_layout(title="Summary Statistics", font=dict(size=18), title_x=0.5, title_y=0.88)

        # Show the table
        # fig.show()

        return fig

    def all(self):

        fig1 = self.line_chart()
        fig2 = self.stats()

        return fig1, fig2