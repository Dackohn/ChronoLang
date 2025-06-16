import React, { useState } from "react";
import "../styles/Work.css";
import { parseCode } from "../../services/api";

const Work = () => {
  const [code, setCode] = useState("");
  const [output, setOutput] = useState({
    text: "",
    table: [],
    structure: null
  });
  const [activeTab, setActiveTab] = useState("text");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleExecute = async () => {
    if (!code.trim()) {
      setError("Please enter ChronoLang code");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const response = await parseCode(code);

      if (response.success && response.result) {
        setOutput({
          text: response.result.text || "",
          table: Array.isArray(response.result.table) ? response.result.table : [],
          structure: response.result.structure || null
        });
      } else {
        throw new Error(response.error || "Execution failed");
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  console.log("structure output", output.structure);

  const renderOutput = () => {
    if (error) {
      return (
        <div className="output-error">
          <h3>Error</h3>
          <pre>{error}</pre>
        </div>
      );
    }

    switch (activeTab) {
      case "text":
        return (
          <div className="output-text">
            <h3>Execution Result</h3>
            <pre>{isLoading ? "Processing..." : output.text || "No output"}</pre>
          </div>
        );
      case "table":
        return (
          <div className="output-table">
            <h3>Table View</h3>
            {output.table.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    {Object.keys(output.table[0]).map(key => (
                      <th key={key}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {output.table.map((row, i) => (
                    <tr key={i}>
                      {Object.values(row).map((val, j) => (
                        <td key={j}>{String(val)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>No table data available</p>
            )}
          </div>
        );
      case "structure": {
  // Support both {results: [...]} and [...] forms
  let plots = [];
  let resultArr = [];

  if (Array.isArray(output.structure)) {
    resultArr = output.structure;
  } else if (
    output.structure &&
    typeof output.structure === "object" &&
    Array.isArray(output.structure.results)
  ) {
    resultArr = output.structure.results;
  }

  plots = resultArr.filter(item => !!item.plot);

  return (
    <div className="output-structure">
      <h3>Plots</h3>
      {plots.length > 0 ? (
        plots.map((item, idx) => (
          <div key={idx} style={{ margin: "16px 0" }}>
            <img
              src={`data:image/png;base64,${item.plot}`}
              alt={`Plot ${idx + 1}`}
              style={{ maxWidth: "100%", border: "1px solid #ccc" }}
            />
            <div style={{ marginTop: "8px", fontStyle: "italic", color: "#333" }}>
              {item.message}
            </div>
          </div>
        ))
      ) : (
        <div>No plots available</div>
      )}
    </div>
  );
}


      default:
        return null;
    }
  };

  return (
    <div className="work-container">
      <h1>ChronoLang Interpreter</h1>

      <div className="editor-container">
        <div className="code-editor">
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            placeholder="Enter your ChronoLang code here..."
            disabled={isLoading}
          />
          <button
            onClick={handleExecute}
            disabled={isLoading || !code.trim()}
          >
            {isLoading ? "Executing..." : "Execute"}
          </button>
        </div>

        <div className="output-section">
          <div className="output-tabs">
            {["text", "table", "structure"].map((tab) => (
              <button
                key={tab}
                className={activeTab === tab ? "active" : ""}
                onClick={() => setActiveTab(tab)}
                disabled={isLoading}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>

          <div className="output-content">
            {renderOutput()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Work;
