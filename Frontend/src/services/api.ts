const API_BASE_URL = import.meta.env.DEV 
  ? '/api' 
  : '/api';  // Use relative path to leverage Vite's proxy

export const parseCode = async (code: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/parse`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ code }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error parsing code:', error);
    throw error;
  }
};
