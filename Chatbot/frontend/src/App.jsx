import Chatbot from "./components/Chatbot";

function App() {
  return (
    <div className="bg-light min-vh-100">
      <nav className="navbar navbar-expand-lg navbar-dark bg-primary">
        <div className="container">
          <span className="navbar-brand mb-0 h1">Interactive Chatbot</span>
        </div>
      </nav>
      <Chatbot />
    </div>
  );
}

export default App;
