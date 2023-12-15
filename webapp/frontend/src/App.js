import { BrowserRouter, Routes, Route } from "react-router-dom";
import NoPage from './pages/nopages';
import Home from './pages/Home';
import Res from "./pages/results";
import About from "./pages/About";
import Generate from "./pages/Generate";


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element = {<About/>} />
        <Route path="results" element = {<Res/>} />
        <Route path="Generate" element = {<Generate/>}/>
        <Route path="About" element = {<About/>}/>
        <Route path="*" element = {<NoPage/>}/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
