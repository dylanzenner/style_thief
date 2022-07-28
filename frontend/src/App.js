import logo from './logo.svg';
import './App.css';
import {useState} from "react";

function App() {
    const [baseImage, setBaseImage] = useState(null);
    const [styleImage, setStyleImage] = useState(null);



    function setBaseImageHandler(event){
        console.log(event.target)
        setBaseImage(URL.createObjectURL(event.target.files[0]))
    }

    function setStyleImageHandler(event){
        console.log(event.target)
        setStyleImage(URL.createObjectURL(event.target.files[0]))
    }

  return (
    <div className="App">
      <div className={'relative text-sky-400 text-5xl mt-4 font-bold'}>
        Style Thief
      </div>

        <div className={'mt-5 grid grid-cols-4 gap-4 content-center'}>

            <div className={'aspect-w-4 aspect-h-3 inline-block ml-5'}>
                <img className={'border-2 rounded'} height={400} width={400} alt={'base image'} src={baseImage}/>

            </div>

            <div className={'aspect-w-4 aspect-h-4 inline-block ml-5'}>
                <img className={'border-2 rounded'} height={400} width={400} alt={'style image'} src={styleImage}/>

            </div>

            <div/>


            <div className={'aspect-w-4 aspect-h-3 inline-block border-2 mr-5'}>
                BEAST
            </div>

        </div>

        <div className={'relative mt-5 grid grid-cols-4 gap-4 content-center'}>

            <div className={'ml-5 hover:-translate-y-0.5'}>
                <label htmlFor="style-image" className={'border-2 cursor-pointer px-2 hover:text-sky-400'}>Upload Base
                    Image</label>
                <input id={'style-image'} type={'file'} className={'hidden'} onChange={setBaseImageHandler}
                       placeholder={''}/>
            </div>

            <div className={'ml-5 hover:-translate-y-0.5'}>
                <label htmlFor="base-image" className={'border-2 cursor-pointer px-2 hover:text-sky-400'}>Upload Style Image</label>
                <input id={'base-image'} type={'file'} className={'hidden'} onChange={setStyleImageHandler}  placeholder={''}/>
            </div>


            <div>
                <button className={'border-2 px-2 hover:-translate-y-0.5 hover:text-sky-400'}>
                    Transfer
                </button>

            </div>

            <div className={'ml-5 hover:-translate-y-0.5'}>
                <label htmlFor="style-image" className={'border-2 cursor-pointer px-2 hover:text-sky-400'}>Download Stylized
                    Image</label>
                <input id={'style-image'} type={'file'} className={'hidden'} onChange={null}
                       placeholder={''}/>
            </div>



        </div>
    </div>
  );
}

export default App;
