'''
This is a demo implementation of building a GPT powered chatbot that can store chat history during the given session.

'''

from openai import OpenAI
import tiktoken

#global variables
summary_model = "gpt-3.5-turbo"
generating_model = "gpt-4o"

#starting OpenAI Session
client = OpenAI()

def check_tokens(text):
    encoder = tiktoken.encoding_for_model(model_name=generating_model)
    tokens = encoder.encode(text)    
    return len(tokens)

def summarize(history):
    #Prompt to generate history
    dialogue = f"Summarize the following dialogue between a user and open AI assistant,\
produce just the summary of the dialogue. \
keep it brief also make sure important details are not missing:\n\n"
    for dic in history:
        #removing unnecessary text like the '{','}','role','content' has shown to use ~25% less tokens
        #These are required when prompting the model, but not necessarily needed during summarisation.

        dialogue += f"{dic['role']}: {dic['content']}\n"

    summary = client.chat.completions.create(
        model= summary_model,
        messages=[{"role":"system", "content": dialogue}],
        max_tokens= 150)
    return summary.choices[0].message.content

def generate(message):
    response = client.chat.completions.create(model=generating_model,messages=message)
    return response.choices[0].message.content, response.usage.completion_tokens, response.usage.prompt_tokens

if __name__ == "__main__":
    chat_history = []
    total_ip_tok = 0
    total_op_tok = 0
    
    print("Session Started... (Type 'exit' to end session)")

    while(True):
        prompt = input("User: ")
        if(prompt == "exit"):
            break

        tokens=check_tokens(str(chat_history))
        if tokens>300:
            #print("summarising....")
            summary=summarize(chat_history)
            chat_history=[dict(role="system", content=f"the following is the summary of exchange between you and user so far:\n"),
                          dict(role="assistant", content=summary),
                          dict(role="user", content=prompt)]
        else:
            chat_history.append(dict(role="user",content= prompt))
        
        response,optk,iptk = generate(chat_history)
        chat_history.append(dict(role="assistant", content=response))
        total_ip_tok += iptk
        total_op_tok += optk

        print(f"\033[96m\nChat: {response}\n\033[0m")

    print(f"Session ended...\nToken Summary:\nInput: {total_ip_tok}\nOutput: {total_op_tok}\nTotal: {total_ip_tok+total_op_tok}")