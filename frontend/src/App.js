import logo from './logo.svg';
import './App.css';
import {useState} from "react";

function App() {
    const [baseImagePreview, setBaseImagePreview] = useState(logo);
    const [baseImage, setBaseImage] = useState(null);
    const [styleImagePreview, setStyleImagePreview] = useState(logo);
    const [styleImage, setStyleImage] = useState(null);



    function setBaseImageHandler(event){
        event.preventDefault()
        setBaseImage(event.target.files[0])
        setBaseImagePreview(URL.createObjectURL(event.target.files[0]))
        
    }

    function setStyleImageHandler(event){
        event.preventDefault()
        setStyleImage(event.target.files[0])
        setStyleImagePreview(URL.createObjectURL(event.target.files[0]))
    }

    function handleOnSubmit(event){
        const formData = new FormData();
        

        formData.append(
            'base_image',
            baseImage,
            baseImage.name
        );

        formData.append(
            'style_image',
            styleImage,
            styleImage.name
        );


        const requestOptions = {
            method: 'POST',
            body: formData
        };
        console.log(requestOptions)


        fetch('http://localhost:8000/upload', requestOptions)
        .then(response => response.json())
        .then(function(response) {
            console.log(response)
        })

    
    }

  return (
    <div className="App">
      <div className={'relative text-sky-400 text-5xl mt-4 font-bold'}>
        Style Thief
      </div>

        <div className={'h-full mt-5 grid grid-cols-4 gap-4 content-center'}>

            <div className={'aspect-w-4 aspect-h-3 inline-block ml-5'}>
                <img className={'border-2 rounded'} height={400} width={400} alt={'base image'} src={baseImagePreview}/>

            </div>

            <div className={'aspect-w-4 aspect-h-4 inline-block ml-5'}>
                <img className={'border-2 rounded'} height={400} width={400} alt={'style image'} src={styleImagePreview}/>

            </div>

            <div/>


            <div className={'aspect-w-4 aspect-h-3 inline-block border-2 mr-5'}>
                BEAST
            </div>

        </div>

        <div className={'relative mt-5 grid grid-cols-4 gap-4 content-center'}>

            <div className={'ml-5'}>                
                <form>
                   <fieldset>
                        <input id={'style-image'} type={'file'} accept={'.jpeg, .png, .jpg'} className={'hidden'} onChange={setBaseImageHandler} placeholder={''}/>
                    </fieldset> 
                    <label htmlFor={'style-image'} className={'border-2 cursor-pointer px-2 hover:text-sky-400'}>Upload Base Image</label>
                </form>
                <br/>
                <button onClick={handleOnSubmit}>Upload</button>
        
            </div>

            <div className={'ml-5'}>
                <form>
                    <fieldset>
                        <input id={'base-image'} type={'file'} accept={'.jpeg, .png, .jpg'} className={'hidden'} onChange={setStyleImageHandler}  placeholder={''}/>
                    </fieldset>
                    <label htmlFor={'base-image'} className={'border-2 cursor-pointer px-2 hover:text-sky-400'}>Upload Style Image</label>
                </form>
            </div>


            <div>
                
                <button onClick={handleOnSubmit} className={'border-2 px-2 hover:-translate-y-0.5 hover:text-sky-400'}>
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
