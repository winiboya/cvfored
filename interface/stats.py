import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class analytics:
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
        temp = df['Frame Number']
        x = temp.values.T.tolist()
        temp = df['Percentages']
        y = temp.values.T.tolist()

        fig = px.line(x=x, y=y, labels={'x': 'Timestamp', 'y': 'Percent Attentive'}, markers=True)

        fig.update_layout(
            title='Attentiveness Over Time', font=dict(size=17), title_x=0.5, title_y=0.97,
            xaxis_title_font=dict(size=18),  # Change font size for the x-axis title
            yaxis_title_font=dict(size=18)   # Change font size for the y-axis title
        )

        # fig.show()

        return fig

    def stats(self):

        df = self.table()

        average = df['Percentages'].mean()

        max = df['Percentages'].max()

        min = df['Percentages'].min()
        
        fig = go.Figure(data=[go.Table(header=dict(values=['', ''], height=30),
                 cells=dict(values=[
            ["Average Attentiveness", "Maximum Attentiveness", "Minimum Attentiveness"],
            [f"{average}%", f"{max}%", f"{min}%"]
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