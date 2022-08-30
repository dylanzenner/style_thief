import logo from './image-placeholder.jpeg';
import './App.css';
import {useState, useEffect} from "react";


function App() {
    const [baseImagePreview, setBaseImagePreview] = useState(logo);
    const [baseImage, setBaseImage] = useState(null);
    const [styleImagePreview, setStyleImagePreview] = useState(logo);
    const [styleImage, setStyleImage] = useState(null);
    const [stylizedImage, setStylizedImage] = useState(logo);
    const [downloadLink, setDownloadLink] = useState(null);
    const [loading, setLoading] = useState(false)


    function setBaseImageHandler(event){
        event.preventDefault()
        setBaseImage(event.target.files[0])
        setBaseImagePreview(URL.createObjectURL(event.target.files[0]))
        console.log(baseImage)
        
    }

    function setStyleImageHandler(event){
        event.preventDefault()
        setStyleImage(event.target.files[0])
        setStyleImagePreview(URL.createObjectURL(event.target.files[0]))
    }

    function handleDownload(event){
        event.preventDefault();
        let link = document.createElement('a');
        link.download = 'style-thief-image.jpeg';
        link.href = downloadLink;
        link.click();
        URL.revokeObjectURL(link.href);
        setStylizedImage(logo);
    }

    useEffect( () => {
        setLoading(false)
        const script = document.createElement('script');
        script.src = "https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js";
        script.async = true;
        document.body.appendChild(script);

    }, [stylizedImage])

    function handleOnSubmit(event){
        setLoading(true)
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
            body: formData,
        };

        fetch('http://localhost:8000/upload', requestOptions)
        .then(response => response.blob())
        .then(blob => {

            let blobURL = URL.createObjectURL(blob);
            let linkURL = window.URL.createObjectURL(blob)
            
            setDownloadLink(linkURL);
            setStylizedImage(blobURL)
            let image = document.getElementById('stylized-image')
            image.onload = function(){
                URL.revokeObjectURL(this.src);
            
            }
            image.src = blobURL;
            setStylizedImage(blobURL)

        })
        .catch(error => {
            console.log(error)
        })
    }


  return (
    <div className="App">
        <div className={""}>
            <div className={'text-5xl mt-4 font-bold text-rose-400'}>
                Style Thief
            </div>
            
                <div className={'mt-5 py-20 grid grid-cols-4 gap-4 content-center'}>

                    <div className={'aspect-w-3 aspect-h-3 inline-block ml-5 bg-white rounded drop-shadow-2xl'}>
                        <img className={'border-2 rounded'} height={400} width={400} alt={'base'} src={baseImagePreview}/>
                        
                    </div>

                    <div className={'aspect-w-3 aspect-h-3 inline-block ml-5 bg-white rounded drop-shadow-2xl'}>
                        <img className={'border-2 rounded'} height={400} width={400} alt={'style'} src={styleImagePreview}/>

                    </div>

                    <div/>
                    {loading === true ? 
                    <div className={'aspect-w-3 aspect-h-3 inline-block border-2 mr-5 bg-white rounded drop-shadow-2xl'}>
                        <span className={"loader inline-block mx-auto my-auto"}></span>
                    </div>
                    :
                    <div className={'aspect-w-3 aspect-h-3 inline-block mr-5 bg-white rounded drop-shadow-2xl'}>
                        <img id={'stylized-image'} className={'border-2 rounded'} height={400} width={400} alt={'stylized'} src={stylizedImage}/>
                    </div>
                    }
                    

                </div>

                <div className={'mt-5 grid grid-cols-4 gap-4 content-center'}>

                    <div className={'ml-5'}>                
                        <form>
                        <fieldset>
                                <input id={'style-image'} type={'file'} accept={'.jpeg, .png, .jpg'} className={'hidden'} onChange={setBaseImageHandler} placeholder={''}/>
                                <label whileHover={{scale: 1.1}} htmlFor={'style-image'} className={'inline-block xs:text-sm md:text-md lg:text-2xl border-2 border-rose-400 bg-white rounded-lg active:bg-gradient-to-r from-rose-100 to-teal-100 drop-shadow-2xl cursor-pointer px-2'}>Upload Base Image</label>

                        </fieldset> 
                        </form>
                    </div>

                    <div className={'ml-5'}>
                        <form>
                            <fieldset>
                                <input id={'base-image'} type={'file'} accept={'.jpeg, .png, .jpg'} className={'hidden'} onChange={setStyleImageHandler}  placeholder={''}/>
                                <label htmlFor={'base-image'} className={'inline-block xs:text-sm md:text-md lg:text-2xl border-2 border-rose-400 cursor-pointer bg-white active:bg-gradient-to-r from-rose-100 to-teal-100 rounded-lg drop-shadow-2xl px-2'}>Upload Style Image</label>

                            </fieldset>
                        </form>
                    </div>


                    <div>
                        
                        <button onClick={handleOnSubmit} className={'xs:text-sm md:text-md lg:text-2xl border-2 border-rose-400 px-2 bg-white active:bg-gradient-to-r from-rose-100 to-teal-100 rounded-lg drop-shadow-2xl'}>
                            Transfer
                        </button>

                    </div>

                    <div id={'stylized-image-button'} className={'mr-5'}>
                        <button className={'xs:text-sm md:text-md lg:text-2xl border-2 border-rose-400 bg-white rounded-lg active:bg-gradient-to-r from-rose-100 to-teal-100 drop-shadow-2xl cursor-pointer px-2'} onClick={handleDownload}>Download Image</button>
                    </div>

                </div>
        </div>
        <br/>
        <br/>
        <div className={"flex"}>
            <h3 className={"xs:text-sm md:text-md lg:text-2xl mx-auto w-48 bg-white border-2 rounded-lg border-rose-400"}>
                Turn your creation into a canvas print!
            </h3>
        </div>

    </div>
  );
}

export default App;
