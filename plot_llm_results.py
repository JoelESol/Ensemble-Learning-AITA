import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class AITAResultsAnalyzer:
    def __init__(self, results_path: str):
        """Initialize analyzer with results file path"""
        self.results_path = Path(results_path)
        self.results_df = None
        self.agent_names = []
        self.verdict_mapping = {
            'asshole': 'ASSHOLE',
            'not the asshole': 'NOT THE ASSHOLE',
            'everyone sucks': 'EVERYONE SUCKS',
            'no assholes here': 'NO ASSHOLES HERE',
            'unknown': 'UNKNOWN',
            'error': 'ERROR'
        }

    def load_results(self) -> pd.DataFrame:
        """Load and process results from JSONL file"""
        print(f"Loading results from {self.results_path}")

        results = []

        with open(self.results_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                base_row = {
                    'id': data['id'],
                    'original_verdict': data.get('original_verdict', 'unknown'),
                    'post_text_preview': data.get('post_text', '')
                }

                # Add each agent's verdict as columns
                for agent, agent_data in data.get('agents', {}).items():
                    base_row[f'{agent}_verdict'] = agent_data.get('verdict', 'unknown')
                    base_row[f'{agent}_explanation'] = agent_data.get('explanation', '')

                results.append(base_row)

        self.results_df = pd.DataFrame(results)

        # Extract agent names from columns (excluding original verdict)
        self.agent_names = [col.replace('_verdict', '') for col in self.results_df.columns
                            if col.endswith('_verdict') and col != 'original_verdict']

        print(f"Loaded {len(self.results_df)} posts with {len(self.agent_names)} agents")
        print(f"Agents: {', '.join(self.agent_names)}")

        return self.results_df

    def normalize_verdict(self, verdict: str) -> str:
        """Normalize verdict strings for comparison"""
        if pd.isna(verdict):
            return 'UNKNOWN'

        verdict_clean = str(verdict).strip().lower()
        return self.verdict_mapping.get(verdict_clean, verdict_clean.upper())

    def calculate_accuracy_by_verdict(self) -> Dict[str, Dict[str, float]]:
        """Calculate accuracy for each agent by verdict type"""
        # Normalize all verdicts
        df = self.results_df.copy()
        df['original_verdict_norm'] = df['original_verdict'].apply(self.normalize_verdict)

        for agent in self.agent_names:
            df[f'{agent}_verdict_norm'] = df[f'{agent}_verdict'].apply(self.normalize_verdict)

        # Calculate accuracy by verdict
        accuracy_by_verdict = {}

        for verdict in df['original_verdict_norm'].unique():
            if verdict in ['UNKNOWN', 'ERROR']:
                continue

            verdict_df = df[df['original_verdict_norm'] == verdict]
            accuracy_by_verdict[verdict] = {}

            for agent in self.agent_names:
                if len(verdict_df) > 0:
                    correct = (verdict_df['original_verdict_norm'] == verdict_df[f'{agent}_verdict_norm']).sum()
                    total = len(verdict_df)
                    accuracy_by_verdict[verdict][agent] = correct / total
                else:
                    accuracy_by_verdict[verdict][agent] = 0.0

        return accuracy_by_verdict

    def calculate_overall_accuracy(self) -> Dict[str, float]:
        """Calculate overall accuracy for each agent"""
        df = self.results_df.copy()
        df['original_verdict_norm'] = df['original_verdict'].apply(self.normalize_verdict)

        # Filter out unknown/error cases
        df = df[~df['original_verdict_norm'].isin(['UNKNOWN', 'ERROR'])]

        overall_accuracy = {}

        for agent in self.agent_names:
            df[f'{agent}_verdict_norm'] = df[f'{agent}_verdict'].apply(self.normalize_verdict)

            correct = (df['original_verdict_norm'] == df[f'{agent}_verdict_norm']).sum()
            total = len(df)

            overall_accuracy[agent] = correct / total if total > 0 else 0.0

        return overall_accuracy

    def create_verdict_accuracy_charts(self, save_path: str = None):
        """Create bar charts showing accuracy by verdict for each agent"""
        accuracy_by_verdict = self.calculate_accuracy_by_verdict()

        if not accuracy_by_verdict:
            print("No verdict data available for visualization")
            return

        # Set up the plot
        verdicts = list(accuracy_by_verdict.keys())
        n_verdicts = len(verdicts)

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agent Accuracy by Verdict Type', fontsize=16, fontweight='bold')

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.agent_names)))

        for i, verdict in enumerate(verdicts):
            if i >= 4:  # Only show first 4 verdicts
                break

            ax = axes[i]

            agents = list(accuracy_by_verdict[verdict].keys())
            accuracies = list(accuracy_by_verdict[verdict].values())

            bars = ax.bar(agents, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

            ax.set_title(f'{verdict}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Hide unused subplots
        for i in range(len(verdicts), 4):
            axes[i].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Verdict accuracy charts saved to: {save_path}")

        plt.show()

    def create_overall_accuracy_chart(self, save_path: str = None):
        """Create bar chart showing overall accuracy for each agent"""
        overall_accuracy = self.calculate_overall_accuracy()

        if not overall_accuracy:
            print("No accuracy data available for visualization")
            return

        # Sort agents by accuracy (descending)
        sorted_agents = sorted(overall_accuracy.items(), key=lambda x: x[1], reverse=True)
        agents, accuracies = zip(*sorted_agents)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create gradient colors based on accuracy
        colors = plt.cm.RdYlGn(np.array(accuracies))

        bars = ax.bar(agents, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_title('Overall Agent Accuracy Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xlabel('Agent', fontsize=14)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add horizontal line at 50% accuracy
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% Baseline')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Overall accuracy chart saved to: {save_path}")

        plt.show()

    def generate_summary_report(self) -> str:
        """Generate a text summary report of the analysis"""
        if self.results_df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        overall_accuracy = self.calculate_overall_accuracy()
        accuracy_by_verdict = self.calculate_accuracy_by_verdict()

        report = []
        report.append("=" * 60)
        report.append("AITA AGENT PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Total posts analyzed: {len(self.results_df)}")
        report.append(f"Number of agents: {len(self.agent_names)}")
        report.append("")

        # Overall accuracy ranking
        report.append("OVERALL ACCURACY RANKING:")
        report.append("-" * 30)
        sorted_agents = sorted(overall_accuracy.items(), key=lambda x: x[1], reverse=True)
        for i, (agent, accuracy) in enumerate(sorted_agents, 1):
            report.append(f"{i}. {agent}: {accuracy:.2%}")
        report.append("")

        # Verdict-specific analysis
        report.append("ACCURACY BY VERDICT:")
        report.append("-" * 30)
        for verdict in accuracy_by_verdict:
            report.append(f"\n{verdict}:")
            verdict_acc = accuracy_by_verdict[verdict]
            sorted_verdict_agents = sorted(verdict_acc.items(), key=lambda x: x[1], reverse=True)
            for agent, accuracy in sorted_verdict_agents:
                report.append(f"  {agent}: {accuracy:.2%}")

        return "\n".join(report)

    def save_detailed_analysis(self, output_path: str = "detailed_analysis.csv"):
        """Save detailed analysis results to CSV"""
        if self.results_df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        # Calculate accuracies
        overall_accuracy = self.calculate_overall_accuracy()
        accuracy_by_verdict = self.calculate_accuracy_by_verdict()

        # Create analysis DataFrame
        analysis_data = []

        for agent in self.agent_names:
            row = {
                'Agent': agent,
                'Overall_Accuracy': overall_accuracy.get(agent, 0)
            }

            # Add verdict-specific accuracies
            for verdict in accuracy_by_verdict:
                row[f'{verdict}_Accuracy'] = accuracy_by_verdict[verdict].get(agent, 0)

            analysis_data.append(row)

        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_csv(output_path, index=False)
        print(f"Detailed analysis saved to: {output_path}")

        return analysis_df


def main():
    """Main execution function"""
    # Initialize analyzer
    analyzer = AITAResultsAnalyzer("dataset/agent_results.jsonl")

    # Load results
    analyzer.load_results()

    # Create output directory
    output_dir = Path("analysis")
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Overall accuracy comparison
    analyzer.create_overall_accuracy_chart(
        save_path=output_dir / "overall_accuracy_comparison.png"
    )

    # 2. Accuracy by verdict charts
    analyzer.create_verdict_accuracy_charts(
        save_path=output_dir / "accuracy_by_verdict.png"
    )

    # 3. Generate and save summary report
    report = analyzer.generate_summary_report()
    print("\n" + report)

    with open(output_dir / "summary_report.txt", "w") as f:
        f.write(report)

    # 4. Save detailed analysis
    analyzer.save_detailed_analysis(output_dir / "detailed_analysis.csv")

    print(f"\nAll analysis outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()