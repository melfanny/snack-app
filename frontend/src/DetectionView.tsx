import { useEffect, useRef, useState } from 'react'

// Secara bawaan, permintaan API memakai proxy Vite menuju backend port 8001.
// VITE_API_URL dapat diisi jika frontend perlu mengakses backend secara langsung.
const API = import.meta.env.VITE_API_URL || ''

type Model = 'baseline' | 'p2'
export type Deteksi = { name: string; score: number; box: number[]; calories: number }
// Data yang dioper ke halaman Estimasi Harga.
export type KirimanDeteksi = {
  foto: Blob
  gambar: string
  deteksi: Deteksi[]
  pilihTerendah?: boolean // true: snack berkalori terendah dipilih lebih dulu
}
type Halaman = 'pick' | 'preview' | 'result'

// Mengelola pemilihan foto, proses deteksi, dan tampilan hasilnya.
// langsung=true: setelah deteksi selesai, hasil kalori dilewati dan langsung memanggil onLanjut.
export default function DetectionView({ onLanjut, langsung = false }: {
  onLanjut?: (data: KirimanDeteksi) => void
  langsung?: boolean
}) {
  const [halaman, setHalaman] = useState<Halaman>('pick')
  const [foto, setFoto] = useState<Blob | null>(null)
  const [gambar, setGambar] = useState('')
  const [daftarDeteksi, setDaftarDeteksi] = useState<Deteksi[]>([])
  const [kameraAktif, setKameraAktif] = useState(false)
  const [pesan, setPesan] = useState('')
  const [sedangMendeteksi, setSedangMendeteksi] = useState(false)
  const [deteksiTerpilih, setDeteksiTerpilih] = useState<number | null>(null)
  const [model, setModel] = useState<Model>('baseline')
  const [modelTerpakai, setModelTerpakai] = useState('baseline')
  const videoRef = useRef<HTMLVideoElement>(null)
  const inputFileRef = useRef<HTMLInputElement>(null)
  const inputKameraRef = useRef<HTMLInputElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Memasang stream kamera setelah elemen video tampil.
  useEffect(() => {
    if (kameraAktif && videoRef.current) {
      videoRef.current.srcObject = streamRef.current
    }
  }, [kameraAktif])

  // Menghentikan semua track kamera yang sedang aktif.
  function hentikanKamera() {
    streamRef.current?.getTracks().forEach(track => track.stop())
    setKameraAktif(false)
  }

  // Menampilkan foto yang dipilih pada halaman pratinjau.
  function tampilkanPratinjau(blob: Blob) {
    setFoto(blob)
    setGambar(URL.createObjectURL(blob))
    setPesan('')
    setHalaman('preview')
  }

  // Menangani foto yang diunggah atau diambil melalui kamera bawaan.
  function tanganiFile(e: React.ChangeEvent<HTMLInputElement>) {
    const fileTerpilih = e.target.files?.[0]
    if (fileTerpilih) {
      hentikanKamera()
      tampilkanPratinjau(fileTerpilih)
    }
    e.target.value = ''
  }

  // Menghapus foto dan kembali ke halaman pilihan foto.
  function ulangi() {
    setFoto(null)
    setGambar('')
    setDaftarDeteksi([])
    setDeteksiTerpilih(null)
    setPesan('')
    setHalaman('pick')
  }

  // Meminta izin dan membuka kamera perangkat.
  async function bukaKamera() {
    if (!navigator.mediaDevices) {
      inputKameraRef.current!.click()
      return
    }

    try {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      })
      setPesan('')
      setKameraAktif(true)
    } catch (error) {
      setPesan(`Kamera gagal dibuka (${error}). Izinkan akses kamera atau gunakan UPLOAD.`)
    }
  }

  // Mengambil satu gambar dari stream kamera yang sedang aktif.
  function ambilFoto() {
    const video = videoRef.current!
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d')!.drawImage(video, 0, 0)
    hentikanKamera()
    canvas.toBlob(blob => blob && tampilkanPratinjau(blob), 'image/jpeg')
  }

  // Membuka kamera atau mengambil foto jika kamera sudah menyala.
  async function tanganiKamera() {
    if (kameraAktif) {
      ambilFoto()
      return
    }
    await bukaKamera()
  }

  // Mengirim foto ke backend dan menampilkan hasil deteksinya.
  async function mulaiDeteksi() {
    if (!foto || sedangMendeteksi) return

    setDaftarDeteksi([])
    setDeteksiTerpilih(null)
    setPesan('')
    setSedangMendeteksi(true)

    const form = new FormData()
    form.append('file', foto, 'snack.jpg')
    form.append('model', model)

    try {
      const respons = await fetch(`${API}/api/detect`, { method: 'POST', body: form })
      if (!respons.ok) throw new Error((await respons.json()).detail)
      const hasil: Deteksi[] = (await respons.json()).detections
      if (langsung && onLanjut) {
        onLanjut({ foto, gambar, deteksi: hasil })
        return
      }
      setDaftarDeteksi(hasil)
      setModelTerpakai(model)
      setPesan('')
      setHalaman('result')
    } catch (error) {
      setPesan(`Error: ${error} (menghubungi ${API}/api/detect)`)
    } finally {
      setSedangMendeteksi(false)
    }
  }

  // Menghapus satu hasil deteksi dari daftar.
  function hapusDeteksi(indeks: number) {
    setDaftarDeteksi(daftarDeteksi.filter((_, indeksDeteksi) => indeksDeteksi !== indeks))
    setDeteksiTerpilih(null)
  }

  const kaloriTerendah = daftarDeteksi.length
    ? Math.min(...daftarDeteksi.map(deteksi => deteksi.calories))
    : 0
  const namaKaloriTerendah = new Set(
    daftarDeteksi
      .filter(deteksi => deteksi.calories === kaloriTerendah)
      .map(deteksi => deteksi.name),
  )

  return (
    <div className={'wrap ' + halaman}>
      <div className="box">
        {halaman === 'pick' && kameraAktif && (
          <video ref={videoRef} autoPlay playsInline muted />
        )}
        {halaman === 'pick' && !kameraAktif && <p>Preview</p>}
        {halaman !== 'pick' && <img src={gambar} />}
        {sedangMendeteksi && (
          <div className="detection-loading" role="status" aria-live="polite">
            <span className="loading-spinner" aria-hidden="true" />
            <b>Sedang menganalisis snack</b>
            <span>Mohon tunggu sebentar...</span>
            <span className="loading-bar" aria-hidden="true"><i /></span>
          </div>
        )}
        {halaman === 'preview' && <b className="x" onClick={ulangi}>✕</b>}
        {halaman === 'result' && daftarDeteksi.map((deteksi, indeks) => (
          <div
            key={indeks}
            onClick={() => setDeteksiTerpilih(indeks)}
            className={
              'bbox'
              + (namaKaloriTerendah.has(deteksi.name) ? ' low' : '')
              + (deteksiTerpilih === indeks ? ' sel' : '')
            }
            style={{
              top: `${deteksi.box[0] * 100}%`,
              left: `${deteksi.box[1] * 100}%`,
              height: `${(deteksi.box[2] - deteksi.box[0]) * 100}%`,
              width: `${(deteksi.box[3] - deteksi.box[1]) * 100}%`,
            }}
          >
            <span>{deteksi.name}</span>
          </div>
        ))}
      </div>

      {halaman !== 'result' && (
        <div className="models">
          <span>Model:</span>
          <button
            disabled={sedangMendeteksi}
            className={model === 'baseline' ? 'active' : ''}
            onClick={() => setModel('baseline')}
          >
            RT-DETR Baseline
          </button>
          <button
            disabled={sedangMendeteksi}
            className={model === 'p2' ? 'active' : ''}
            onClick={() => setModel('p2')}
          >
            RT-DETR P2
          </button>
        </div>
      )}

      <div className="row">
        {halaman === 'pick' && (
          <>
            <button onClick={tanganiKamera}>{kameraAktif ? 'CAPTURE' : 'TAKE'}</button>
            <button onClick={() => inputFileRef.current!.click()}>UPLOAD</button>
          </>
        )}
        {halaman === 'preview' && (
          <>
            <button
              className="primary"
              disabled={sedangMendeteksi}
              onClick={mulaiDeteksi}
            >
              {sedangMendeteksi ? 'MEMPROSES...' : 'START'}
            </button>
            <button disabled={sedangMendeteksi} onClick={ulangi}>UPLOAD ULANG</button>
          </>
        )}
        {halaman === 'result' && <button onClick={ulangi}>FOTO BARU</button>}
        {halaman === 'result' && onLanjut && (
          <button
            className="primary"
            onClick={() => onLanjut({ foto: foto!, gambar, deteksi: daftarDeteksi, pilihTerendah: true })}
          >
            Lanjut ke Price Estimation
          </button>
        )}
        <input
          ref={inputFileRef}
          type="file"
          accept="image/*"
          hidden
          onChange={tanganiFile}
        />
        <input
          ref={inputKameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          hidden
          onChange={tanganiFile}
        />
      </div>

      {pesan && <p role="alert">{pesan}</p>}
      {halaman === 'result' && (
        <p className="used">
          Model: <b>{modelTerpakai === 'p2' ? 'RT-DETR P2' : 'RT-DETR Baseline'}</b>
        </p>
      )}
      {halaman === 'result' && daftarDeteksi.length === 0 && (
        <p>Tidak ada snack terdeteksi.</p>
      )}
      {halaman === 'result' && daftarDeteksi.length > 0 && (
        <div className="list">
          {daftarDeteksi.map((deteksi, indeks) => (
            <div
              key={indeks}
              onClick={() => setDeteksiTerpilih(indeks)}
              className={
                'item'
                + (namaKaloriTerendah.has(deteksi.name) ? ' low' : '')
                + (deteksiTerpilih === indeks ? ' sel' : '')
              }
            >
              <span className="name">
                {deteksi.name} {namaKaloriTerendah.has(deteksi.name) && '✅ TERENDAH'}
              </span>
              <span className="cal">{deteksi.calories} kkal/100 g</span>
              <b
                className="del"
                title="Hapus"
                onClick={e => {
                  e.stopPropagation()
                  hapusDeteksi(indeks)
                }}
              >
                🗑️
              </b>
            </div>
          ))}
        </div>
      )}
      {halaman === 'result' && (
        <div className="note">
          <p>ℹ️ <b>Info:</b> Sistem bisa saja salah mendeteksi. Cocokkan kembali dengan kemasan produk.</p>
          <p>🥗 <b>Tips sehat:</b> Nilai kalori di atas per 100 g. Pilih snack berkalori lebih rendah, batasi porsi, dan imbangi dengan buah, sayur, serta air putih.</p>
        </div>
      )}
    </div>
  )
}
