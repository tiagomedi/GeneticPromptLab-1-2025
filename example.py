import pandas as pd
import json
from GeneticPromptLab import QuestionsAnswersOptimizer

def trec():
    # Configuration
    train_path = './data/trec_train.csv'
    test_path = './data/trec_test.csv'
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open("./data/trec_label_dict.json", "r") as f:
        label_dict = json.load(f)
        label_dict = {i:v for i,v in enumerate(label_dict)}
    problem_description = "Data are collected from four sources: 4,500 English questions. Your objective is to classify these into one of the following labels: "+str(label_dict)

    train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    # Create GeneticPromptLab instance
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def agnews():
    # Configuration
    train_path = './data/ag_news_train.csv'
    test_path = './data/ag_news_test.csv'
    model_name = 'multi-qa-MiniLM-L6-cos-v1'
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    with open("./data/ag_news_label_dict.json", "r") as f:
        label_dict = json.load(f)
        label_dict = {i:v for i,v in enumerate(label_dict)}
    problem_description = "AG is a collection of more than 1 million news articles. News articles have been gathered from more than 2000 news sources by ComeToMyHead in more than 1 year of activity. ComeToMyHead is an academic news search engine which has been running since July, 2004. The dataset is provided by the academic comunity for research purposes in data mining. Your objective is a classification label, with possible values including World (0), Sports (1), Business (2), Sci/Tech (3)."

    train_questions_list, train_answers_label, test_questions_list, test_answers_label = train_data['question'].tolist(), train_data['label'].tolist(), test_data['question'].tolist(), test_data['label'].tolist()
    # Create GeneticPromptLab instance
    return problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name

def main():
    print("🚀 GeneticPromptLab - Sistema Solo SSH/Ollama")
    print("=" * 60)
    
    print("\n📋 AGNEWS Classification:")
    problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name = agnews()
    population_size = 8
    generations = 10
    sample_p = 0.01
    num_retries = 2

    lab = QuestionsAnswersOptimizer(
        problem_description=problem_description, 
        train_questions_list=train_questions_list, 
        train_answers_label=train_answers_label, 
        test_questions_list=test_questions_list, 
        test_answers_label=test_answers_label, 
        label_dict=label_dict, 
        model_name=model_name, 
        sample_p=sample_p, 
        init_and_fitness_sample=population_size, 
        window_size_init=2,
        num_retries=num_retries,
        ssh_credentials="ssh_credentials.json",
        modelo_llm="llama3.1",
        temperatura=0.7
    )
    
    print(f"✅ Optimizador AG News configurado - usando SSH/Ollama")
    optimized_prompts = lab.genetic_algorithm(generations)
    print(f"📝 Prompts optimizados:")
    for i, prompt in enumerate(optimized_prompts[:3]):
        print(f"  {i+1}. {prompt[:100]}...")
    
    print("\n" + "="*60)
    print("📋 TREC Classification:")
    problem_description, train_questions_list, train_answers_label, test_questions_list, test_answers_label, label_dict, model_name = trec()
    population_size = 8
    generations = 10
    sample_p = 0.01
    num_retries = 2

    lab = QuestionsAnswersOptimizer(
        problem_description=problem_description, 
        train_questions_list=train_questions_list, 
        train_answers_label=train_answers_label, 
        test_questions_list=test_questions_list, 
        test_answers_label=test_answers_label, 
        label_dict=label_dict, 
        model_name=model_name, 
        sample_p=sample_p, 
        init_and_fitness_sample=population_size, 
        window_size_init=2, 
        num_retries=num_retries,
        ssh_credentials="ssh_credentials.json",
        modelo_llm="llama3.1",
        temperatura=0.7
    )
    
    print(f"✅ Optimizador TREC configurado - usando SSH/Ollama")
    optimized_prompts = lab.genetic_algorithm(generations)
    print(f"📝 Prompts optimizados:")
    for i, prompt in enumerate(optimized_prompts[:3]):
        print(f"  {i+1}. {prompt[:100]}...")
    
    print("\n🎉 -------- EXPERIMENTOS COMPLETADOS --------")
    print("💾 Resultados guardados en directorio 'runs/'")
    print("📊 Usa visualizer.py para ver gráficos de evolución")

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        print("\n💡 Asegúrate de que:")
        print("   - ssh_credentials.json esté configurado")
        print("   - El servidor SSH esté accesible")
        print("   - Ollama esté ejecutándose con llama3.1")
        print("   - Los archivos de datos estén en ./data/")