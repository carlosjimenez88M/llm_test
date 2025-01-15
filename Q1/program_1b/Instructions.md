# InsuranceRAG: A Smart Insurance Query System 🤖

This project implements a Retrieval-Augmented Generation (RAG) system for handling insurance-related queries. Think of it as your highly knowledgeable insurance assistant that learns from existing documentation.

## Why InsuranceRAG? 🎯

Ever tried explaining insurance policies to customers? It's like teaching a cat to swim - technically possible but usually messy. This system:

- Handles repetitive insurance queries accurately
- Tracks its own performance (because who doesn't love metrics?)




### Prerequisites 📋

You'll need:
- Python 3.10+
- An OpenAI API key (export it as an environment variable)
- MLflow and Weights & Biases accounts

```bash
# Set your OpenAI API key
export OPENAI_API_KEY='abcdrfg'
```

## Running the System 🚀

It's as simple as:

```bash
mlflow run .
```

This will:
1. Set up the vector store from your reference data
2. Initialize the RAG agent
3. Process test queries
4. Generate evaluation reports

## Project Structure 📁

```
.
├── configs/
│   └── config.yaml         
├── data/
│   ├── reference_data.csv   
│   ├── current_data.csv     
│   └── interaction_log.json 
├── modules/                 
└── reports/                 
```

## Where to Find Your Results 📊

After running the system, check out:

1. **Evaluation Reports**: 
   - Location: `reports/evaluation_report.txt`
   - Contains: Precision, recall, F1 scores, and response similarity metrics
   - Pro tip: This is your go-to for performance analysis

2. **Interaction Logs**:
   - Location: `data/interaction_log.json`
   - Tracks: Every query-response pair
   - Use case: Perfect for debugging weird responses

3. **MLflow & W&B Dashboards**:
   - Access: Through your MLflow/W&B accounts
   - Shows: Real-time metrics and experiment tracking
   - Best for: Comparing different runs and configurations

## Customization 🎨

Want to tweak the system? Edit `configs/config.yaml`:

```yaml
paths:
  reference_data: data/reference_data.csv  
  current_data: data/current_data.csv      
  logs: data/interaction_log.json          
  drift_report: reports/evaluation_report.txt  
```

## Performance Metrics 📈

The system tracks:
- **Precision**: How accurate are the responses?
- **Recall**: Are we covering all the important points?
- **F1 Score**: Balance between precision and recall
- **Response Similarity**: How close are we to ground truth?
- **Perfect Matches**: When we nail it completely

## Contributing 🤝

Found a bug? Want to add a feature? We're all ears! Just:
1. Fork the repo
2. Create your feature branch
3. Send us a pull request

## Notes 📝

- Keep your `reference_data.csv` clean and well-formatted
- The system learns from your data - garbage in, garbage out
- Monitor the evaluation reports regularly


## Need Help? 🆘

- Check the evaluation reports first
- Review the interaction logs
- Still stuck? Open an issue!

