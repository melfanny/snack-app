// Daftar merek untuk halaman informasi; label kelas model dan CSV tetap sama.
const BRANDS = [
  { brand: 'Chitato', items: ['Aburi', 'Bumbu Sapi', 'Sapi Panggang'] },
  { brand: 'Chitato Lite', items: ['Ayam Bawang', 'Rumput Laut', 'Salmon', 'Saus Bawang'] },
  { brand: 'Japota', items: ['Madu', 'Rumput Laut'] },
  { brand: 'Potabee', items: ['Ayam Bakar', 'Rumput Laut', 'Sapi Panggang', 'Spicy'] },
  { brand: 'Qtela', items: ['Balado', 'Original'] },
]

// Menampilkan petunjuk penggunaan dan daftar snack yang didukung.
export default function TutorialView() {
  return (
    <div className="card">
      <h3>How to use</h3>
      <ol>
        <li>Tekan TAKE untuk mengambil foto langsung menggunakan kamera (pada perangkat seluler, aplikasi kamera bawaan akan terbuka), atau tekan UPLOAD untuk memilih foto dari galeri.</li>
        <li>Foto yang dipilih akan tampil sebagai pratinjau. Tekan ✕ atau UPLOAD ULANG untuk menggantinya dengan foto lain.</li>
        <li>Sebelum menekan START, pilih model deteksi: RT-DETR Baseline atau RT-DETR P2. Model yang dipilih akan digunakan untuk memproses foto tersebut; Anda bebas mencoba keduanya karena hasil deteksinya bisa berbeda.</li>
        <li>Tekan START untuk memulai proses deteksi. Pada halaman hasil, produk snack yang ditandai dengan warna hijau merupakan produk dengan kandungan kalori terendah. Model yang digunakan juga ditampilkan pada halaman hasil.</li>
      </ol>

      <h3>About</h3>
      <p>Aplikasi ini mendeteksi produk snack menggunakan model RT-DETRv2 (PyTorch) dengan backbone HGNetv2-L dalam dua varian: Baseline dan P2 (menggunakan modul S2-CCFF untuk fitur objek kecil), dengan tingkat kepercayaan (confidence) minimal 0,50 kemudian menampilkan informasi kalori per 100 g untuk setiap produk yang terdeteksi.</p>

      <p><b>Jenis snack yang dapat dideteksi (15 kelas):</b></p>
      {BRANDS.map(b => (
        <div key={b.brand} className="brand-group">
          <div className="brand-title">{b.brand}</div>
          <div className="chips">
            {b.items.map(i => <span key={i} className="chip">{i}</span>)}
          </div>
        </div>
      ))}
    </div>
  )
}
