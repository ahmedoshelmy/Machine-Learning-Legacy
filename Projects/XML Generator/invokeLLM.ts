import { ChatGroq } from "@langchain/groq";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { v4 as uuidv4 } from 'uuid';
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
} from "@langchain/langgraph";
import dotenv from 'dotenv';
dotenv.config();

/**
 * Function to invoke an LLM with a given prompt.
 * @param prompt - The prompt to be sent to the LLM.
 * @returns The LLM's response as a string.
 */
// Function to create the LLM
function createLLM(): ChatGroq {
  return new ChatGroq({
    model: "llama3-70b-8192",
    temperature: 0.1,
    apiKey: ""
  });
}

// Function to create the prompt template
function createPromptTemplate(): ChatPromptTemplate {
  const prompt = `
      You are a diagram designer and developer specializing in creating XML diagrams that are compatible with Draw.io.

      Your task is to:
      - Generate accurate and fully-renderable XML code for diagrams, ensuring compatibility with Draw.io.
      - Follow the Draw.io XML structure and syntax strictly.
      - Respond with XML code only: no explanations, formatting characters, or additional text of any kind.

      For consistency and accuracy, ensure:
      - The XML is syntactically valid and complete.
      - Elements and attributes conform to Draw.io standards for rendering.
      - If the provided input includes specific requirements (e.g., shapes, connections, layout), incorporate them exactly as described.

      You must produce well-formed XML directly compatible with Draw.io, without any extraneous characters or commentary.
      `;

  return ChatPromptTemplate.fromMessages([
    [
      "system",
      prompt
    ],
    new MessagesPlaceholder("messages"),
  ]);
}

export class LLMInstance {
  private app: any;
  private config: any;
  private memory: MemorySaver;

  constructor() {
    this.memory = new MemorySaver();
    const workflow = this.setupWorkflow();
    this.app = this.configureApp(workflow);
    this.config = { configurable: { thread_id: uuidv4() } };  // Unique thread ID for each instance
  }

  // Set up the workflow with the LLM node
  private setupWorkflow(): any {
    return new StateGraph(MessagesAnnotation)
      .addNode("model", this.invokeLLM.bind(this))
      .addEdge(START, "model")
      .addEdge("model", END);
  }

  // Configure the app with the given workflow
  private configureApp(workflow: any): any {
    return workflow.compile({ checkpointer: this.memory });
  }

  // Function to invoke LLM with a given state
  private async invokeLLM(state: typeof MessagesAnnotation.State): Promise<any> {
    try {
      const llm = createLLM();
      const promptTemplate = createPromptTemplate();
      const chain = promptTemplate.pipe(llm);

      const response = await chain.invoke(state);
      return { messages: [response] };
    } catch (error) {
      console.error("Error invoking LLM:", error);
      throw error;
    }
  }

  // Prepare input with the user's prompt
  private prepareInput(prompt: string): any[] {
    return [
      {
        role: "user",
        content: prompt,
      },
    ];
  }

  // Call the LLM with a prompt and store in memory
  public async callLLM(prompt: string): Promise<string> {
    const input = this.prepareInput(prompt);
    const output = await this.app.invoke({ messages: input }, this.config);

    console.log(output.messages[output.messages.length - 1].content);

    return output.messages[output.messages.length - 1].content;
  }

  // Regenerate XML from the last call stored in memory
  public async regenerateXML(): Promise<string> {
    const regeneratedResponse = await this.app.invoke(
      { messages: "Regenerate the last xml code you produced." },
      this.config
    );
    console.log(regeneratedResponse.messages[regeneratedResponse.messages.length - 1].content);

    return regeneratedResponse.messages[regeneratedResponse.messages.length - 1].content;
  }
}