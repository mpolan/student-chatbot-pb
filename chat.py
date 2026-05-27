import ollama

model_name = "llama3.2" #"mistral"
messages = [
    {'role': 'system', "content": "Jesteś pomocnym bot'em. Odpowiadaj miło i w języku polskim."},
]

while True:
    prompt = input("Ty: ")
    if not prompt:
        break

    messages.append(
        {"role": "user", "content": prompt}
    )

    response = ollama.chat(
        model=model_name, 
        messages=messages, 
        #tools=[testowa_funkcja]
    )
    # if response.message.tool_calls:
    #     for tool_call in response.message.tool_calls:
    #         args = tool_call.function.arguments
    #         result = testowa_funkcja(**args)
    #         messages.append({"role": "tool", "content": str(result)})
        
    #     response = ollama.chat(model=model_name, messages=messages)

    answer = response.message.content
    messages.append({"role": "assistant", "content": answer})
    print("Bot: ", answer)