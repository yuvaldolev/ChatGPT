import { OpenAIModel, OpenAIModelID, OpenAIModels } from "@/types/openai";

const api = async (key: string): Promise<Response> => {
  try {
    const response = await fetch("https://api.openai.com/v1/models", {
      headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${key ? key : process.env.OPENAI_API_KEY}`
    }
    });

    if (response.status !== 200) {
      throw new Error("OpenAI API returned an error");
    }

    const json = await response.json();

    const models: OpenAIModel[] = json.data
    .map((model: any) => {
      for (const [key, value] of Object.entries(OpenAIModelID)) {
        if (value === model.id) {
          return {
            id: model.id,
            name: OpenAIModels[value].name
          };
        }
      }
    })
    .filter(Boolean);

    return new Response(JSON.stringify(models), { status: 200 });
  } catch (error) {
    console.error(error);
    return new Response("Error", { status: 500 });
  }
};

export default api;
